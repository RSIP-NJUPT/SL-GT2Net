# model settings
# norm_cfg = dict(type="BN", requires_grad=True)
# conv_cfg = dict(type="Conv3d")
# norm_cfg3d = dict(type="BN3d", requires_grad=True)
# patch_size = [2, 4, 4]
# img_size = [12, 224, 224]
# block_cls='ISABlock' # todo TwinsBlock
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        type="SLGT2",
        img_size=[16, 224, 224],
        patch_size=[2, 4, 4],
        in_chans=2,
        num_classes=8,
        wss_lst=[[7, 14, 21], [7, 14, 21], [7, 14, 14], [7, 7, 7]],
        dss_lst=[[8, 4, 2], [4, 4, 2], [2, 2, 2], [1, 1, 1]],
        topk_div=2,
        num_heads=[2, 4, 8, 16],
        embed_dims=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,  # todo 0.2
        norm_layer="nn.LayerNorm",
        depths=[2, 2, 6, 2],
        layer_cls="SLGT2Layer",
        layer_scale=None,
        ape=False,
        rel_pos=False,
    ),
    decode_head=dict(
        type="PseudoHead", num_classes=8, loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)
    ),
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode="whole")
