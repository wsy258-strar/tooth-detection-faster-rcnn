_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',  # Faster R-CNN 基础配置
    '../_base_/datasets/coco_detection.py',     # 数据集配置
    '../_base_/schedules/schedule_1x.py',       # 学习率调度配置
    '../_base_/default_runtime.py'              # 运行时配置
]

# Swin Transformer 预训练权重
pretrained = '/data/wangshunyi/mmdetection/pre-trained-models/swin_small_patch4_window7_224.pth'

# 模型配置
model = dict(
    type='FasterRCNN',  # 修改为 FasterRCNN
    backbone=dict(
        _delete_=True,  # 删除原有的 ResNet backbone 配置
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768]  # 与 Swin Transformer 输出通道匹配
    ),
    roi_head = dict(
        bbox_head = dict(
            num_classes = 32,
            # reg_decoded_bbox=True,
            # loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
        )
    ),
)

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
        weight_decay=0.05
    )
)

dataset_type = 'CocoDataset'
data_root = '/data/wangshunyi/mmdetection/data/DENTEX/new_enumeration_data/'
classes = (
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '41', '42', '43', '44', '45', '46', '47', '48',
    '31', '32', '33', '34', '35', '36', '37', '38'
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset = dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/')
    )
)

val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = dict(ann_file=data_root + 'test.json')

train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=max_epochs,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次

val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

# load_from = '/data/wangshunyi/mmdetection/pre-trained-models/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # 每1个epoch保存一次
        save_best='coco/bbox_mAP',  # 根据COCO的mAP@[0.5:0.95]保存最佳模型
        rule='greater',  # 指标越大越好
        max_keep_ckpts=5  # 仅保留最好的一个检查点
    )
)