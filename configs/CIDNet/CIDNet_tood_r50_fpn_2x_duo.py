# base：引入其他配置文件，以便复用数据集、训练计划和运行时设置，确保配置的模块化和可重用性。
_base_ = [
    '../_base_/datasets/duo_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='TOOD', # type='TOOD'：指定模型为TOOD（Task-Oriented Object Detection）类型。
    ################################################################################

    #####################################################################

    backbone=dict(
        type='ResNet', # 选择ResNet作为背骨网络。
        depth=101, # 使用50层的ResNet。
        num_stages=4, # num_stages=4：网络分为4个阶段提取特征。
        out_indices=(0, 1, 2, 3), # out_indices=(0, 1, 2, 3)：表示从每个阶段输出特征图。
        frozen_stages=1, # frozen_stages=1：冻结第1阶段的参数以减少计算量。
        norm_cfg=dict(type='BN', requires_grad=True), # 使用批量归一化，并在训练中更新其参数。
        norm_eval=True, # 在评估阶段使用批量归一化的评估模式。
        style='pytorch',
        # 使用预训练的ResNet-50权重进行初始化。
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
    ),
    neck=dict(
        type='CIDNet',
        in_channels=[256, 512, 1024, 2048], # 定义来自不同阶段的输入特征图的通道数。
        out_channels=256), # 输出特征图的通道数。

    bbox_head=dict(
        type='TOODHead', # 选择TOOD头作为边框预测模块。
        num_classes=4, # 目标检测任务中目标的类别数。
        in_channels=256, # 来自neck层输出的特征图的通道数。
        stacked_convs=6, # 使用的卷积层数量。
        feat_channels=256, # 特征通道数，指边框头卷积层的输出通道数。
        anchor_type='anchor_free', # 表示不使用传统锚框，而是采用无锚框策略。
        anchor_generator=dict( # 配置锚框生成器。
            type='AnchorGenerator', # 选择锚框生成器类型。
            ratios=[1.0], # 定义锚框的宽高比。
            octave_base_scale=8, # 基础尺度，用于生成不同尺度的锚框。
            scales_per_octave=1, # 每个八度音阶的锚框数量。
            strides=[8, 16, 32, 64, 128]), # 不同层级的特征图对应的步幅。
        # bbox_coder=dict(
        #     type='DeltaXYWHBBoxCoder', # type='DeltaXYWHBBoxCoder'：选择编码器类型。
        #     target_means=[.0, .0, .0, .0], # 目标框中心坐标和尺寸的均值，用于归一化。
        #     target_stds=[0.1, 0.1, 0.2, 0.2]), # 目标框中心坐标和尺寸的标准差，用于归一化。
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',  # type='DeltaXYWHBBoxCoder'：选择编码器类型。
            target_means=[.0, .0, .0, .0],  # 目标框中心坐标和尺寸的均值，用于归一化。
            target_stds=[0.05, 0.05, 0.1, 0.1]),  # 目标框中心坐标和尺寸的标准差，用于归一化。
        initial_loss_cls=dict( # 初始分类损失配置。
            type='FocalLoss', # 选择Focal Loss，适用于处理类别不平衡的问题。
            use_sigmoid=True, # 使用sigmoid激活函数。
            activated=True,  # use probability instead of logit as input 输入为概率而非logit。
            gamma=2.0, # Focal Loss的超参数。
            alpha=0.25,
            loss_weight=1.0), # 损失权重。
        loss_cls=dict(
            type='QualityFocalLoss', # 使用质量Focal Loss，考虑样本的质量。
            use_sigmoid=True, #
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4, # initial_epoch=4：从第4个周期开始训练。
        initial_assigner=dict(type='ATSSAssigner', topk=9), # ：初始分配器使用ATSS方法，设置topk为9。
        assigner=dict(type='TaskAlignedAssigner', topk=13), # 后续的分配器设置为TaskAligned，topk为13。
        # alpha=1、beta=6：超参数，调节正负样本的权重。
        alpha=1,
        beta=6,
        # allowed_border=-1：允许的边框范围，-1表示无边界限制。
        allowed_border=-1,
        # pos_weight=-1：正样本的权重，-1表示自动计算。
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000, # 在非极大值抑制前，最多预留1000个候选框。
        min_bbox_size=0, # 最小边框大小，0表示没有限制。
        score_thr=0.05, # 得分阈值，低于此值的预测框将被丢弃。
        nms=dict(type='nms', iou_threshold=0.6), # 非极大值抑制设置，使用IoU阈值0.6来过滤重叠框。
        max_per_img=100)) # 每张图像最多保留100个预测框。
# optimizer
# type='SGD'：选择随机梯度下降（SGD）作为优化器。
# lr=0.001：设置学习率为0.001。
# momentum=0.9：设置动量为0.9，以加速收敛。
# weight_decay=0.0001：设置权重衰减，防止过拟合。
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# custom hooks
# custom_hooks：自定义钩子，用于在训练过程中执行特定的操作。
# type='SetEpochInfoHook'：设置一个钩子，用于在每个周期结束时更新相关信息。
custom_hooks = [dict(type='SetEpochInfoHook')]

# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.1,
#     step=[30, 40])
# # runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=50)
# evaluation = dict(interval=2)
