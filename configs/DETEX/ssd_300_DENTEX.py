_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# export CUBLAS_WORKSPACE_CONFIG=:4096:8

randomness = dict(
    seed = 2023,
    diff_rank_seed=True,
    deterministic=True
)

dataset_type = 'CocoDataset'
data_root = '/data/wangshunyi/mmdetection/data/DENTEX/new_enumeration_data/'
classes = (
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48'
)

# dataset settings
input_size = 512
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
    type='AutoAugment',
    policies=[
        [dict(type='Contrast',level=7,prob=1.0)],]
    ),
    dict(
        type='Expand',
        mean={{_base_.model.data_preprocessor.mean}},
        to_rgb={{_base_.model.data_preprocessor.bgr_to_rgb}},
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    # dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type={{_base_.dataset_type}},
            data_root=data_root,
            metainfo=dict(classes=classes),
            ann_file=data_root +'train.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args={{_base_.backend_args}})))

val_dataloader = dict(
        batch_size=2, 
        dataset=dict(
            type=dataset_type,
            test_mode=True,
            metainfo=dict(classes=classes),
            data_root=data_root,
            ann_file=data_root + 'val.json',
            data_prefix=dict(img='val/'), 
        ) 
)
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=data_root + 'test.json',
        data_prefix=dict(img='test/')
    )
)

val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'val.json',
                     metric='bbox',
                     format_only=False,)
test_evaluator = dict(ann_file=data_root + 'test.json')

model = dict(
    neck=dict(
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        last_kernel_size=4),
    bbox_head=dict(
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        num_classes=32,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]))
)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=3e-3, momentum=0.9, weight_decay=5e-4))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=100,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # 每1个epoch保存一次
        save_best='coco/bbox_mAP',  # 根据COCO的mAP@[0.5:0.95]保存最佳模型
        rule='greater',  # 指标越大越好
        max_keep_ckpts=5  # 仅保留最好的一个检查点
    )
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
