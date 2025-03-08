# dataset settings
dataset_type = "BrandenburgDataset"
data_root = r"data/brandenburg/s1_2017_2018"
# print(data_root)
# two channels, i.e., dual polarization (VV+VH)
background = (255,255,255) #BGR
img_norm_cfg_trainval = dict(mean=[-16.37, -10.59], std=[3.12, 3.20], to_rgb=False)
img_norm_cfg_test = dict(mean=[-16.37, -10.58], std=[3.12, 3.21], to_rgb=False)

num_bands = 48
img_scale = (224, 224)
crop_size = (224, 224)
train_pipeline = [
    dict(type="LoadDataFromFile", field_name="data"),
    dict(type="LoadAnnotations", field_name="label", reduce_zero_label=True),
    dict(type="Resize", img_scale=img_scale, ratio_range=(1.0, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg_trainval),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255, num_bands=num_bands, mode="constant"), # todo
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadDataFromFile", field_name="data"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),  # todo
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg_trainval),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255, num_bands=num_bands), # todo
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadDataFromFile", field_name="data"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),  # todo
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg_test),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255, num_bands=num_bands), # todo
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# 224
data = dict(
    samples_per_gpu=2,  # batch size
    workers_per_gpu=8,  # num_workers
    # samples_per_gpu=1,  # batch size
    # workers_per_gpu=4,  # num_workers
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="trainval_vhvv",
        ann_dir="trainval_label",
        split="split/train.txt",
        pipeline=train_pipeline,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=True,  # ignore label 0
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="trainval_vhvv",
        ann_dir="trainval_label",
        split="split/val.txt",
        pipeline=val_pipeline,
        test_mode=True,
        ignore_index=255,
        reduce_zero_label=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="test_vhvv",
        ann_dir="test_label",
        split="split/test.txt",
        pipeline=test_pipeline,
        test_mode=True,
        ignore_index=255,
        reduce_zero_label=True,
    ),
)
