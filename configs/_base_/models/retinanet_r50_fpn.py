# model settings
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet', # 调用mmdet/models/backbones/resnet.py
        depth=50,
        # ResNet 系列包括 stem+ 4个 stage 输出
        num_stages=4, 
        # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要，
        # 其 stride 为 (4,8,16,32)，输出的channel 为 (256, 512, 1024, 2048)
        # stride表示模型的下采样率，假设图片输入是320x320，stride=10,那么输出特征图是32x32的，假设每个位置anchor是5个，那么这个输出特征图就一共有32x32x5和anchor
        out_indices=(0, 1, 2, 3),
        # 表示固定 stem 加上第一个 stage 的权重，不进行训练
        frozen_stages=1,
        # 所有的 BN 层的可学习参数都需要梯度，BN里的/gamma和/beta两个可学习参数，这两个参数是BN里的可学习参数，参与网络反向传播的过程。
        norm_cfg=dict(type='BN', requires_grad=True),
        # norm_eval=True是指BN层不计算也不更新均值和方差，要用时我直接使用训练好的均值和方差
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN', #调用 mmdet/models/necks/fpn.py
        # ResNet 模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN 输出的每个尺度输出特征图通道
        out_channels=256,
        # 从输入多尺度特征图的第1个开始计算（0,1,2,3,)
        start_level=1,
        # 额外输出层的特征图来源
        add_extra_convs='on_input', # 额外输出的 2 个特征图的来源是骨架网络输出，而不是 FPN 层本身输出又作为后面层的输入
        # FPN 输出特征图个数
        num_outs=5), #  ResNet 输出 4 个不同尺度特征图 (c2,c3,c4,c5)，stride 分别是 (4,8,16,32)，通道数为 (256,512,1024,2048)
        # FPN网络流程
        # 将 c3、c4 和 c5 三个特征图全部经过各自 1x1 卷积进行通道变换得到 m3~m5，输出通道统一为 256
        # 从 m5(特征图最小)开始，先进行 2 倍最近邻上采样，然后和 m4 进行 add 操作，得到新的 m4
        # 将新 m4 进行 2 倍最近邻上采样，然后和 m3 进行 add 操作，得到新的 m3
        # 对 m5 和新融合后的 m4、m3，都进行各自的 3x3 卷积，得到 3 个尺度的最终输出 P5～P3
        # 将 c5 进行 3x3 且 stride=2 的卷积操作，得到 P6
        # 将 P6 再一次进行 3x3 且 stride=2 的卷积操作，得到 P7
        # P6 和 P7 目的是提供一个大感受野强语义的特征图，有利于大物体和超大物体检测。 在 RetinaNet 的 FPN 模块中只包括卷积，不包括 BN 和 ReLU。

        # 大stride → 小特征图：大stride会导致特征图尺寸缩小，每个特征点对应原图的更大区域，感受野大，适合捕捉大物体的全局信息。
        # 小stride → 大特征图：小stride保留了高分辨率特征图，避免小物体在下采样过程中丢失细节，保留细节，适合捕捉小物体的局部特征。
    bbox_head=dict(
        type='RetinaHead', # 位置为mmdet/models/dense_heads/retina_head.py
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict( #3种尺度3中比例一共九种anchor
            type='AnchorGenerator',
            # 特征图 anchor 的 base scale, 值越大，所有 anchor 的尺度都会变大，如果自定义数据集中普遍都是大物体或者小物体，则可能修改更改 octave_base_scale 参数
            octave_base_scale=4,
            # 每个特征图有3个尺度，2**0, 2**(1/3), 2**(2/3)，2**(1/3)表示对原图乘2**(1/3)
            scales_per_octave=3,
            # 每个特征图有3个高宽比例 1:2 1:1 2:1
            ratios=[0.5, 1.0, 2.0],
            # 特征图对应的 stride，必须特征图 stride 一致，不可以随意更改
            strides=[8, 16, 32, 64, 128]),
        # BBox Encoder Decoder
        # 1.更好的平衡分类和回归分支 loss，以及平衡 bbox 四个预测值的 loss
        # 2.训练过程中引入 anchor 信息，加快收敛    
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder', # 代码在mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]), 
        # 分类loss
        loss_cls=dict(
            type='FocalLoss', # 代码在mmdet/models/losses/focal_loss.py
            use_sigmoid=True,
            # gamma 有focal效应，可以控制难易样本权重，值越大，对分类错误样本梯度越大(难样本权重大)，focal 效应越大，这个参数非常关键。
            gamma=2.0,
            # alpha属于正负样本的加权参数，值越大，正样本的权重越大
            alpha=0.25,
            loss_weight=1.0),
        # 回归loss
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        # BBox Assigner：
        # 如果 anchor 和所有 gt bbox 的最大 iou 值小于 0.4，那么该 anchor 就是背景样本
        # 如果 anchor 和所有 gt bbox 的最大 iou 值大于等于 0.5，那么该 anchor 就是高质量正样本
        # 如果 gt bbox 和所有 anchor 的最大 iou 值大于等于 0(确保了每个真实目标框至少有一个锚点与之匹配)，那么该 gt bbox 所对应的 anchor 也是正样本
        # 其余样本全部为忽略样本即 anchor 和所有 gt bbox 的最大 iou 值处于 [0.4,0.5) 区间的 anchor 为忽略样本，不计算 loss
        assigner=dict(
            # 最大 IoU 原则分配器
            type='MaxIoUAssigner',  # 调用mmdet/models/task_modules/assigners/max_iou_assigner.py
            # 正样本阈值
            pos_iou_thr=0.5,
            # 负样本阈值
            neg_iou_thr=0.4,
            # 正样本阈值下限
            min_pos_iou=0,
            # 忽略 bboxs 的阈值，-1表示不忽略
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        # nms 前每个输出层最多保留1000个预测框
        nms_pre=1000,  #后处理NMS代码在 mmdet/models/dense_heads/base_dense_head.py
        # 过滤掉的最小 bbox 尺寸
        min_bbox_size=0,
        # 分值阈值
        score_thr=0.05,
        # nms 方法和 nms 阈值
        nms=dict(type='nms', iou_threshold=0.5),
        # 最终输出的每张图片最多 bbox 个数
        max_per_img=100))
