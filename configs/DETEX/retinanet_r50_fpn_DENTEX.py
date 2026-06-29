_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../retinanet/retinanet_tta.py'
]

randomness = dict(
    seed = 2023,
    diff_rank_seed=True,
    deterministic=True
)
# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

# 训练配置
max_epochs = 100
# train_cfg = dict(max_epochs=max_epochs)

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=1000
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1
    )
]

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    ),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.04
    )
)

dataset_type = 'CocoDataset'
data_root = '/data/wangshunyi/mmdetection/data/DENTEX/new_enumeration_data/'
classes = (
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48'
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset = dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=data_root +'train.json',
        data_prefix=dict(img='train/'),
    )
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
    bbox_head=dict(
        type='RetinaHead',
        num_classes=32,  # 将类别数从80改为32
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
)

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