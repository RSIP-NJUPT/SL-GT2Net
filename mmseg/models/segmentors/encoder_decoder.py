import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .base import BaseSegmentor
from .. import builder
from ..builder import SEGMENTORS


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained) # todo
        self.decode_head.init_weights() # todo
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)

        out = self._decode_head_forward_test(x, img_metas)
        # todo
        # if out.ndim == 5:
        #     outs = [resize(
        #             input=logit,
        #             size=img.shape[-2:],
        #             mode='bilinear',
        #             align_corners=self.align_corners) for logit in out]
        #     out = torch.stack(outs, dim=0)
        # else: # out.ndim==4
        out = resize(
            input=out,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images. BxCxHxW
            sequence_imgs (Tensor): Input sequence_imgs. BxTxCxHxW
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        num_classes = self.num_classes
        batch_size, _, d_img, h_img, w_img = img.size()
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1 # todo
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                if img.ndim == 5: # B,C,D,H,W
                    crop_img = img[:, :, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    # todo
                    preds += F.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
                else: # img.ndim == 4, B,C,H,W
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale: # todo
            preds = resize(
                    preds,
                    size=img_meta[0]['ori_shape'][1:3],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            # if img.ndim == 5:
            #     # todo resize C,D,H,W, not B,C,H,W.
            #     preds = [resize(
            #     pred,
            #     size=img_meta[0]['ori_shape'][1:3], # D,H,W,C
            #     mode='bilinear',
            #     align_corners=self.align_corners,
            #     warning=False) for pred in preds]
            #     preds = torch.stack(preds, dim=0)
            # else:
            #     preds = resize(
            #         preds,
            #         size=img_meta[0]['ori_shape'][:2],
            #         mode='bilinear',
            #         align_corners=self.align_corners,
            #         warning=False)
        return preds
    
    # todo
    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][1:3],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    # todo 
    def whole_inference_test(self, img, img_meta, rescale, shape):
        """Inference with full image."""
        img = img[:, :, :shape[0], :shape[1]]
        # sequence_imgs = sequence_imgs[:, :, :, :shape[0], :shape[1]]
        # print('\n test_fps: ', img.shape, sequence_imgs.shape)
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            sequence_imgs (Tensor): The input sequence img of shape (B, T, 3, H, W)
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole', 'test']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        elif self.test_cfg.mode == 'test':
            seg_logit = self.whole_inference_test(img, img_meta, rescale, self.test_cfg.shape)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal': # todo
                if output.ndim == 5:
                    output = output.flip(dim=(4,))
                else:
                    output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                if output.ndim == 4:
                    output = output.flip(dim=(3,))
                else:
                    output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
