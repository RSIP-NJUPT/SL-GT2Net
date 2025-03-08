# dataset settings
dataset_type = "SloveniaDataset"
data_root = "data/slovenia"
# two channels, i.e., dual polarization (VV+VH)
img_norm_cfg = dict(mean=[127.09, 52.14], std=[43.30, 40.76], to_rgb=False)
background = (0,0,0) # BGR
num_bands = 16
img_scale = (500, 500)
crop_size = (224, 224)
train_pipeline = [
    dict(type="LoadDataFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255, num_bands=num_bands, mode="constant"), # todo
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadDataFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),  # todo
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255, num_bands=num_bands), # todo
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = val_pipeline
data = dict(
    samples_per_gpu=2,  # batch size
    workers_per_gpu=8,  # num_workers
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="s1/train",
        ann_dir="label/train",
        pipeline=train_pipeline,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=True,  # ignore label 0
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="s1/val",
        ann_dir="label/val",
        pipeline=val_pipeline,
        test_mode=True,
        ignore_index=255,
        reduce_zero_label=True,
    ),  # ignore label 0
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="s1/test",
        ann_dir="label/test",
        pipeline=test_pipeline,
        test_mode=True,
        ignore_index=255,
        reduce_zero_label=True,
    ),
)  # ignore label 0
