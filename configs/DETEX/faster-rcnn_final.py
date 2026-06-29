_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# _base_ = [
#     '../_base_/models/faster-rcnn_r50_fpn.py',
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]

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

pretrained = '/data/wangshunyi/mmdetection/pre-trained-models/swin_small_patch4_window7_224.pth'
# 不继承基类的train_pipeline和test_pipeline，将基类pipeline注释，自己在文件中实现
backend_args =  dict(_delete_=True,backend='matplotlib')            # None
img_scale = (2000,1200)
train_pipeline = [
    dict(type='LoadImageFromFile',backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    # type='AutoAugment',
    # policies=[
    #     [dict(type='Contrast',level=10,prob=0.5)],
    #     [dict(type='Brightness',level = 7,prob =0.5)],
    #     [dict(type='Equalize',level = 8,prob=1.0)],
    #     [dict(type="RandomErasing",n_patches=(1,3),ratio=(0.02,0.05))]
    #     ]
    # ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.8, 1.4),
        saturation_range=(0.7, 1.3),
        hue_delta=4
    ),
    
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset = dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        pipeline = train_pipeline
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
test_evaluator = dict(ann_file=data_root + 'test.json',
                      format_only=True,
                      outfile_prefix='./work_dirs/coco_detection/test'
                      )

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
            loss_bbox=dict(
                type='CustomSmoothL1Loss',
                loss_weight=1.0, 
                crown_weight=1.0, 
                root_weight=2.5
            )
        )
    ),
    rpn_head = dict(
        anchor_generator = dict(
            scales=[8], # Faster R-CNN 的 RPN 中减小锚点尺寸（例如将默认的 [8, 16, 32] 调整为 [4, 8, 16]），以适应小牙齿。 50 epoch AP3.16
            ratios=[0.21, 0.34, 0.57, 0.77, 0.8], 
            strides=[4, 8, 16, 32, 64]
        ),
        loss_bbox=dict(
            type='CustomSmoothL1Loss',
            loss_weight=1.0, 
            crown_weight=1.0, 
            root_weight=2.5
        )
    ),
    test_cfg=dict(
    # 如何获取ROI，调用了test过程
        rpn=dict( #代码在mmdet/models/dense_heads/base_dense_head.py _predict_by_feat_single中进行后处理操作
            # 是否跨层进行 NMS 操作
            nms_across_levels=False,
            # nms 前每个输出层最多保留 1000 个预测框
            nms_pre=1000,
            # nms 后每张图片最多保留 1000 个预测框
            max_per_img=1000,
            # nms 阈值
            nms=dict(type='nms', iou_threshold=0.7),
            # 过滤掉的最小 bbox 尺寸
            min_bbox_size=0),
            # 经过 RPN test 计算后每张图片可以提供最多 max_per_img 个候选框，一般该值为 2000。
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=32)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
)

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
        weight_decay=0.04
    )
)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001))


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

visualization=dict( #用户可视化验证和测试结果
    type='DetVisualizationHook',
    draw=True,
    interval=1,
    show=False)

