optimizer = dict(type="AdamW", lr=0.0001, weight_decay=1e-5)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=0, by_epoch=False)
# lr_config = dict(warmup='linear', warmup_iters=2000,
#                   warmup_by_epoch=False, policy='CosineAnnealing', min_lr=0, by_epoch=False)
# runtime settings
interval = 2000
runner = dict(type="IterBasedRunner", max_iters=40000)
checkpoint_config = dict(
    by_epoch=False,
    interval=interval,
    max_keep_ckpts=3,
    create_symlink=False,
    save_optimizer=True,
)
evaluation = dict(interval=interval, metric=["mIoU", "mF1score"])  # nan_to_num=0
find_unused_parameters = True
