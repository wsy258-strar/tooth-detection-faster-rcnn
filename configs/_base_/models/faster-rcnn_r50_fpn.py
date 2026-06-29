# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        # ResNet 系列包括 stem+ 4个 stage 输出
        num_stages=4,
        # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要，
        # 其 stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        out_indices=(0, 1, 2, 3),
        # 表示固定 stem 加上第一个 stage 的权重，不进行训练
        frozen_stages=1,
        # 所有的 BN 层的可学习参数都不需要梯度，也就不会进行参数更新
        norm_cfg=dict(type='BN', requires_grad=True),
        # backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        # ResNet 模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN 输出的每个尺度输出特征图通道
        out_channels=256,
        num_outs=5),
        # faster-rcnn FPN网络流程

        # 和 RetinaNet 的区别是 ResNet 输出的 4 个特征图都会被利用。其详细流程是：
        # 将c2 c3 c4 c5 4 个特征图全部经过各自 1x1 卷积进行通道变换变成 m2~m5，输出通道统一为 256
        # 从 m5 开始，先进行 2 倍最近邻上采样，然后和 m4 进行 add 操作，得到新的 m4
        # 将新 m4 进行 2 倍最近邻上采样，然后和 m3 进行 add 操作，得到新的 m3
        # 将新 m3 进行 2 倍最近邻上采样，然后和 m2 进行 add 操作，得到新的 m2
        # 对 m5 和新的融合后的 m4 ~ m2，都进行各自的 3x3 卷积，得到 4 个尺度的最终输出 p5 ~ p2
        # 将 c5 进行 3x3 且 stride=2 的卷积操作，得到 p6，目的是提供一个感受野非常大的特征图，有利于检测超大物体
        # 故 FPN 模块实现了c2 ~ c5 4 个特征图输入，p2 ~ p6 5个特征图输出，其 strides = (4,8,16,32,64)。

        # Strides（步长） 指特征图相对于输入图像的下采样倍数，
        # Stride = 输入图像尺寸 / 特征图尺寸
        # 即特征图上每个像素点对应原始输入图像的区域大小（以像素为单位）。
        # 它反映了特征图的空间分辨率和感受野范围，是理解多尺度目标检测的关键参数。
    rpn_head=dict(
        # FPN 输出的 5 个特征图，输入到同一个 RPN 或者说 5 个相同的 RPN 中，每个特征图都连接一个RPN网络，
        # 每个分支都进行前后景分类和 bbox 回归，5 个 RPN Head 共享所有分类或者回归分支的卷积权重，经过 Head 模块的前向流程输出一共是 5*2 个特征图。
        type='RPNHead',  # mmdet/models/dense_heads/rpn_head.py
        # FPN 层输入特征图通道数
        in_channels=256,
        # 中间特征图通道数
        feat_channels=256,
        # 相比不包括 FPN 的 Faster R-CNN 算法，由于其 RPN Head 是多尺度特征图，为了适应这种变化，
        # anchor 设置进行了适当修改，FPN 输出的多尺度信息可以帮助区分不同大小物体识别问题，每一层就不再需要不包括 FPN 的 Faster R-CNN 算法那么多 anchor 了。

        # 可以看出一共 5 个输出层，每个输出层包括 3 个高宽比例和 1 种尺度，也就是说每一层的每个特征图坐标处都包括 3 个 anchor，一共 15 个 anchor
        anchor_generator=dict( 
            type='AnchorGenerator',
            # 相当于 octave_base_scale，表示每个特征图的 base scales
            scales=[8],
            # 每个特征图有 3 个高宽比例
            ratios= [0.5, 1.0, 2.0],
            # 特征图对应的 stride，必须和特征图 stride 一致，不可以随意更改
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

        # R-CNN 模块接收 RPN 输出的每张图片共 max_per_img 个候选框，然后对这些候选框进一步 refine，
        # 输出包括区分具体类别和 bbox 回归。该模块网络构建方面虽然简单，但是也包括了 RPN 中涉及到的组件，
        # 例如 BBox Assigner、BBox Sampler、BBox Encoder Decoder、Loss 等等，除此之外，还包括一个额外的 RPN 到 R-CNN 数据转换模块：RoIAlign 或者 RoIPool
    roi_head=dict(
        # 一次 refine head，另外对应的是级联结构
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor', # 代码在mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            # 2 个共享 FC 模块
            type='Shared2FCBBoxHead',
            # 输入通道数，相等于 FPN 输出通道
            in_channels=256,
            # 中间 FC 层节点个数
            fc_out_channels=1024,
            # RoIAlign 或 RoIPool 输出的特征图大小
            roi_feat_size=7,
            # 类别个数
            num_classes=80,
            # bbox 编解码策略，除了参数外和 RPN 相同，
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            # 影响 bbox 分支的通道数，True 表示 4 通道输出，False 表示 4×num_classes 通道输出
            reg_class_agnostic=False,
            # CE Loss
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # L1 Loss
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
            # RPN 层输出每张图片最多 max_per_img 个候选框，故 R-CNN 输入 shape 为 (batch, max_per_img, 4)，4 表示 RoI 坐标
            # 利用 RoI 重映射规则，将 nms_pomax_per_imgst 个候选框映射到 FPN 输出的不同特征图上，提取对应的特征图，然后利用插值思想将其变成指定的固定大小输出，
            # 输出 shape 为 (batch, max_per_img, 256, roi_feat_size, roi_feat_size)，其中 256 是 FPN 层输出特征图通道大小，roi_feat_size 一般取 7。
            # 上述步骤即为 RoIAlign 或者 RoIPool 计算过程
            # 将 (batch, max_per_img, 256, roi_feat_size, roi_feat_size) 数据拉伸为 (batch*max_per_img, 256*roi_feat_size*roi_feat_size)，
            # 转化为 FC 可以支持的格式, 然后应用两次共享卷积，输出 shape 为 (batch*nms_pomax_per_imgst, 1024)
            # 将 (batch*max_per_img, 1024) 分成分类和回归分支，分类分支输出 (batch*max_per_img, num_classes+1), 回归分支输出 (batch*max_per_img, 4*num_class)
            # 第二步的映射规则是在 FPN 论文中提出。不知大家是否有疑问：假设某个 proposal 是由第 4 个 特征图层检测出来的，
            # 为啥该 proposal 不是直接去对应特征图层切割就行，还需要重新映射？原因是这些 proposal 是 RPN 测试阶段检测出来的，
            # 大部分 proposal 可能符合前面设定，但是也有很多不符合的，也就是说测试阶段上述一致性不一定满足，需要重新映射，映射公式在知乎中
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                # 最大 IoU 原则分配器
                type='MaxIoUAssigner',
                # 正样本阈值
                pos_iou_thr=0.7,
                # 负样本阈值
                neg_iou_thr=0.3,
                # 正样本阈值下限
                min_pos_iou=0.3,
                # 适当增加更多正样本
                match_low_quality=True,
                # 忽略 bboxes 的阈值，-1 表示不忽略 
                ignore_iof_thr=-1),
                # 如果 anchor 和所有 gt bbox 的最大 iou 值小于 0.3，那么该 anchor 就是背景样本
                # 如果 anchor 和所有 gt bbox 的最大 iou 值大于等于 0.7，那么该 anchor 就是高质量正样本，该阈值比较高，这个阈值设置需要和后续的 R-CNN 模块匹配
                # 如果 gt bbox 和所有 anchor 的最大 iou 值大于等于 0.3(可以看出可能有某些 gt bbox 没有和任意 anchor 匹配)，那么该 gt bbox 所对应的 anchor 也是正样本
                # 其余样本全部为忽略样本，但是由于 neg_iou_thr 和 min_pos_iou 相等，故不存在忽略样本
            # 和 RetinaNet 采用 Focal Loss 处理正负样本不平衡不同，Faster R-CNN 是通过正负样本采样模块来克服。
            sampler=dict(
                # 经过随机采样函数后，可以有效控制 RPN 网络计算 loss 时正负样本平衡问题
                type='RandomSampler',  # 代码在mmdet/models/task_modules/samplers/random_sampler.py 采样正负样本在基类BaseSampler.py实现
                # 采样后每张图片的训练样本（Anchors）总数，不包括忽略样本
                num=256,
                # 正样本比例
                pos_fraction=0.5,
                # 正负样本比例，用于确定负样本采样个数上界
                neg_pos_ub=-1,
                # 是否加入 gt 作为 proposals 以增加高质量正样本数
                add_gt_as_proposals=False),
                # num = 256 表示采样后每张图片的样本总数，pos_fraction 表示其中的正样本比例，具体是正样本采样 128 个，那么理论上负样本采样也是 128 个
                # neg_pos_ub 表示负和正样本比例上限，用于确定负样本采样个数上界，例如打算采样 1000 个样本，正样本打算采样 500 个，但是可能正样本才 200 个，
                # 那么正样本实际上只能采样 200 个，如果设置 neg_pos_ub=-1 那么就会对负样本采样 800 个，用于凑足 1000 个，但是如果设置了 neg_pos_ub 比例，
                # 例如 1.5，那么负样本最多采样 200x1.5=300 个，最终返回的样本实际上不够 1000 个，默认情况 neg_pos_ub=-1
                # add_gt_as_proposals=True 是防止高质量正样本太少而加入的，可以保证前期收敛更快、更稳定，属于训练技巧，在 RPN 模块设置为 False，
                # 主要用于 R-CNN，因为前期 RPN 提供的正样本不够，可能会导致训练不稳定或者前期收敛慢的问题。
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                # 和 RPN 一样，正负样本定义参数不同
                type='MaxIoUAssigner',
                # 正样本阈值
                # 3 个 iou_thr 设置都是 0.5，不存在忽略样本，这个参数在 Cascade R-CNN 论文中有详细说明，影响较大
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                # 为了避免出现低质量匹配情况(因为 two-stage 算法性能核心在于 R-CNN，RPN 主要保证高召回率，R-CNN 保证高精度)，R-CNN 阶段禁用了允许低质量匹配设置
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                # 和 RPN 一样，随机采样参数不同
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                # True，RPN 中为 False
                # add_gt_as_proposals=True。主要是克服刚开始 R-CNN 训练不稳定情况
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
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
            max_per_img=50)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
