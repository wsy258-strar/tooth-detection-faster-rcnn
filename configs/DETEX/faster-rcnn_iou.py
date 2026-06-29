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


# 不继承基类的train_pipeline和test_pipeline，将基类pipeline注释，自己在文件中实现
# backend_args = None
# train_pipeline = [
    # dict(type='LoadImageFromFile',backend_args=backend_args),
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    # type='AutoAugment',
    # policies=[
    #     [dict(type='Contrast',level=10,prob=1.0)],
    #     ]
    # ),
    # dict(_delete_ = True,type='Contrast', prob=1.0, level=10,max_mag = 190),
    # dict(type='Brightness', prob=1.0, level=10),
    # dict(type='AutoAugment', policies=policies_v0()),  # 增强操作
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 生成scale_factor
    # dict(
    #     type='RandomResize',
    #     scale=[(2000, 1200)],
    #     keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PackDetInputs')
# ]
# test_pipeline = [
    # dict(type='LoadImageFromFile',backend_args=backend_args),
    # dict(type='LoadAnnotations', with_bbox=True),1
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='PackDetInputs',
    #     meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
    #                 'scale_factor'))
# ]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset = dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train.json',
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
        ann_file='val.json',
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
        ann_file='test.json',
        data_prefix=dict(img='test/')
    )
)

val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = dict(ann_file=data_root + 'test.json')

model = dict(
    # backbone=dict(
    #     _delete_ = True,
    #     type='ResNeXt',
    #     depth=101,
    #     groups=64,
    #     base_width=4,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     style='pytorch',
    #     init_cfg=dict(
    #         type='Pretrained', checkpoint='/data/wangshunyi/mmdetection/pre-trained-models/resnext101_64x4d-ee2c6f71.pth')),
    rpn_head=dict(
        # reg_decoded_bbox=True, 
        # loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)
        loss_bbox=dict(
                    type='CustomSmoothL1Loss',
                    loss_weight=1.0, 
                    crown_weight=1.0, 
                    root_weight=2.0
        )
    ),
    roi_head = dict(
        bbox_head = dict(
            num_classes = 32,

            loss_bbox=dict(
                    type='CustomSmoothL1Loss',
                    loss_weight=1.0, 
                    crown_weight=1.0, 
                    root_weight=2.0
            )
            # reg_decoded_bbox=True,
            # loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)
        )
    ),

    # rpn_head = dict(
    #     anchor_generator = dict(
    #         scales=[8], # Faster R-CNN 的 RPN 中减小锚点尺寸（例如将默认的 [8, 16, 32] 调整为 [4, 8, 16]），以适应小牙齿。 50 epoch AP3.16
    #         ratios=[0.25, 0.3, 0.5],
    #         strides=[4, 8, 16, 32, 64]
    #     )
    # ),

#     test_cfg=dict(
#         # 如何获取ROI，调用了test过程
#         rpn=dict( #代码在mmdet/models/dense_heads/base_dense_head.py _predict_by_feat_single中进行后处理操作
#             # 是否跨层进行 NMS 操作
#             nms_across_levels=False,
#             # nms 前每个输出层最多保留 1000 个预测框
#             nms_pre=1000,
#             # nms 后每张图片最多保留 1000 个预测框
#             max_per_img=1000,
#             # nms 阈值
#             nms=dict(type='nms', iou_threshold=0.7),
#             # 过滤掉的最小 bbox 尺寸
#             min_bbox_size=0),
#             # 经过 RPN test 计算后每张图片可以提供最多 max_per_img 个候选框，一般该值为 2000。
#         rcnn=dict(
#             score_thr=0.05,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100)
#     )
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

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001))
# load_from = '/data/wangshunyi/mmdetection/pre-trained-models/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

