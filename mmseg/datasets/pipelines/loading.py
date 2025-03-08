import os.path as osp
import scipy.io as sio
import numpy as np

import mmcv
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadDataFromFile(object):
    """Load an data from .mat file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(
        self,
        field_name="band",
        to_float32=False,
    ):
        self.to_float32 = to_float32
        self.field_name = field_name

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results.get("img_prefix") is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        # t h w c for slovenia and brandenburg
        # t c h w for pastis-r
        if filename.endswith(".mat"):  # for slovenia or brandenburg
            img = sio.loadmat(filename)[self.field_name]
            # if "slovenia" in results['img_prefix'].lower(): # for slovenia, t=12 h=500 w=500 c=2
            #     pass
            # elif 'brandenburg' in results['img_prefix'].lower(): # for brandenburg, t=41 h=224 w=224 c=2
            #     pass # todo====  how to deal with t=41?
        elif filename.endswith(".npy"):
            img = np.load(filename)  # t c h w for pastis-r
            # if 'pastis' in results['img_prefix'].lower(): # for pastis-r, t=[65,69,70,71] c=3 h=128 w=128
            #     # todo how to deal with t? note that we need t h w c!!! padding!!!
            #     if img.shape[0] != 71:
            #         num_padding = 71 - img.shape[0]
            #         img = np.pad(img, ((num_padding//2, num_padding - num_padding//2), (0,0), (0,0), (0,0)))
            #         assert img.shape[0] == 71
            img = img.transpose((0, 2, 3, 1))  # t h w c
        else:
            raise NotImplementedError

        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[-1]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32},"
        repr_str += f"field_name='{self.field_name}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):  # todo
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(
        self,
        field_name="label",
        reduce_zero_label=False,
        file_client_args=dict(backend="disk"),
        imdecode_backend="pillow",
    ):
        self.reduce_zero_label = reduce_zero_label
        self.field_name = field_name

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]

        # height = width = 500
        if filename.endswith(".mat"):  # for slovenia or brandenburg
            gt_semantic_seg = sio.loadmat(filename)[self.field_name]  # H,W
        elif filename.endswith(".npy"):
            gt_semantic_seg = np.load(filename)  # C=3,H,W
            if "pastis" in results["seg_prefix"].lower():  # for pastis-r
                gt_semantic_seg = gt_semantic_seg[0]
        gt_semantic_seg = gt_semantic_seg.astype(np.uint8)
        # modify if custom classes
        # if results.get('label_map', None) is not None:
        #     for old_id, new_id in results['label_map'].items():
        #         gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label

        if self.reduce_zero_label:  # todo
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label},"
        repr_str += f"field_name='{self.field_name}')"
        return repr_str
