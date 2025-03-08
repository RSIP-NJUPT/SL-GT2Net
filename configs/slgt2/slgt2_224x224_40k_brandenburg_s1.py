_base_ = [
    "../_base_/models/slgt2.py",
    "../_base_/datasets/brandenburg_s1_224x224.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
num_classes = 15
# model settings
class_weight = None
model = dict(
    backbone=dict(
        type="SLGT2",
        num_classes=num_classes,
        flag=2,
        topk_div=1, # todo
        img_size=[48, 224, 224],  # padding 41 to 48
        patch_size=[2, 4, 4],
        embed_dims=[96, 192, 384, 768],
        wss_lst=[[7, 14, 21], [7, 14, 21], [7, 14, 14], [7, 7, 7]],
        dss_lst=[[8, 4, 2], [8, 4, 2], [4, 4, 2], [2, 2, 2]],
        # ms=2, i.e., 7-14
        # wss_lst=[[7, 14], [7,14], [7, 14], [7,7]],
        # dss_lst=[[8,4], [8,4], [4,4], [2,2]],
        # ms=1, i.e., 7
        # wss_lst=[[7,], [7,], [7,], [7,]],
        # dss_lst=[[8,], [8,], [4,], [2,]],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 6, 2],
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        rel_pos=False,
    ),
    decode_head=dict(
        type="PseudoHead",
        num_classes=num_classes,
        ignore_index=255,  # default
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            use_focal_loss=False,  # use fl
            use_fl_cel=False,  # use fl and cel
            loss_weight=1.0,
            class_weight=class_weight,
        ),
    ),
)

test_cfg = dict(mode='slide', crop_size=(224, 224), stride=(200, 200))
# test_cfg = dict(mode="slide", crop_size=(128, 128), stride=(128, 128))
