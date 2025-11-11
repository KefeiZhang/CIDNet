# Copyright (c) OpenMMLab. All rights reserved. 版权声明，说明该代码归 OpenMMLab 所有。
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
# 从上级目录导入 BACKBONES 构建器和 ResLayer 工具。
from ..builder import BACKBONES
from ..utils import ResLayer

#有用
class BasicBlock(BaseModule):
    expansion = 1
# 构造函数，初始化 BasicBlock。参数包括：
    def __init__(self,
                 inplanes, # 输入通道数。
                 planes, # 输出通道数。
                 stride=1, # 步幅，默认值为 1。
                 dilation=1, # 膨胀率，默认为 1。
                 downsample=None, # 下采样层，默认值为 None。
                 style='pytorch',
                 with_cp=False, # 是否使用检查点，默认为 False。
                 conv_cfg=None, # 卷积层的配置。
                 norm_cfg=dict(type='BN'), # 归一化层的配置，默认为批归一化（BN）。
                 dcn=None, # 可选的可变形卷积，默认为 None。
                 plugins=None, # 可选的插件，默认为 None。
                 init_cfg=None): # 初始化配置。
        super(BasicBlock, self).__init__(init_cfg)
        # 检查 dcn 和 plugins 参数是否为 None，如果不为 None，则抛出错误，表明尚未实现这些功能。
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        # 根据归一化配置创建两个归一化层，并生成对应的名称。
        # norm_cfg: 这是一个配置字典，通常包含了归一化层的类型（如 Batch Normalization、Layer Normalization 等）和其他相关参数。
        # planes: 这是一个整数，代表当前层的输出通道数（即特征图的深度）。
        # postfix=1: 这是一个可选参数，通常用于给创建的层命名或区分不同的层。这里的 postfix 值为 1，表示这是第一层的归一化。
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        # 使用指定的卷积配置构建第一个卷积层。
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        # 将第一个归一化层添加到模块中。
        self.add_module(self.norm1_name, norm1)
        # 构建第二个卷积层，输入和输出通道相同，卷积核大小为 3。
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        # 将第二个归一化层添加到模块中。
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True) # 定义 ReLU 激活函数，设置 inplace=True 可节省内存。
        self.downsample = downsample # 保存下采样层，用于处理输入的捷径连接。
        # 保存步幅、膨胀率和检查点标志。
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    # 定义属性 norm1，用于获取第一个归一化层。
    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name) # 返回动态获取的第一个归一化层

    # 定义属性 norm2，用于获取第二个归一化层。
    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name) # 返回动态获取的第二个归一化层。

    def forward(self, x):  # 定义前向传播函数，接收输入 x。
        """Forward function."""

        def _inner_forward(x): # 定义内部前向传播函数，处理实际的计算。

            identity = x    # 保存输入 x 用于后续的捷径连接

            # 执行第一个卷积、归一化和激活操作。
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            # 执行第二个卷积和归一化操作。
            out = self.conv2(out)
            out = self.norm2(out)

            # 如果存在下采样层，对输入 x 进行下采样。
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity # 定义前向传播函数，接收输入 x。

            return out # 返回经过残差连接后的输出。
        # 如果开启了检查点且 x 需要梯度，使用检查点机制进行前向传播，以节省内存。 下面是另一种解释
        # 如果启用了检查点并且输入 x 需要梯度，则使用 cp.checkpoint 来执行 _inner_forward，以节省内存。
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else: # 否则，直接调用内部前向函数。
            out = _inner_forward(x)

        out = self.relu(out) # 对最终的输出 out 再次应用 ReLU 激活函数，以确保输出非负。

        return out
# 定义一个 Bottleneck 类，继承自 BaseModule。这个类用于构建 ResNet 的瓶颈块。
# expansion 属性定义了该块的输出通道数是输入通道数的 4 倍。
class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes, # inplanes 和 planes：输入和输出通道数。
                 planes,
                 stride=1, # 卷积层的步幅。
                 dilation=1, # 卷积膨胀率。
                 downsample=None, # 用于下采样的模块。
                 style='pytorch',
                 with_cp=False, # 是否启用检查点（减少内存使用）。
                 # 用于配置卷积层和归一化层。
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None, # 可变形卷积的配置。
                 plugins=None, # 插件列表，用于在不同卷积层之后增加插件。
                 init_cfg=None): # 初始化配置，用于模型的权重初始化。
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg) # 调用父类的构造函数，初始化模块。
        # 检查 style 是否为 'pytorch' 或 'caffe'，检查 dcn 是否为 None 或字典类型，检查 plugins 是否为 None 或列表类型。
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        # 如果有插件，则确保插件的 position 字段是合法的，允许的值是 after_conv1、after_conv2 和 after_conv3。
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes # inplanes 和 planes：输入和输出通道数。
        self.planes = planes
        self.stride = stride # 卷积层的步幅。
        # dilation 表示卷积的膨胀率（也称为扩张率）。膨胀卷积可以在不增加计算量的情况下扩大卷积核的感受野。
        # dilation=1 表示普通卷积，值越大，感受野越大。
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn # 可变形卷积的配置。
        self.with_dcn = dcn is not None # 布尔值，表示是否使用可变形卷积。
        self.plugins = plugins
        self.with_plugins = plugins is not None # 布尔值，表示是否有插件。

        if self.with_plugins: # 这一行判断是否有插件。如果 self.with_plugins 为 True，表示在网络中配置了插件，接下来就会处理这些插件。
            # collect plugins for conv1/conv2/conv3
            # 这一行将插件列表中的插件提取出来，并保存到 self.after_conv1_plugins 中。
            # 具体提取的插件是那些位置（position）为 'after_conv1' 的插件。
            # plugin['position'] == 'after_conv1': 这是一个过滤条件，表示只提取位置位于 'after_conv1' 的插件。
            # 也就是说，这些插件会插入到网络的第一个卷积层之后。
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        # 根据 style 来设置步幅：
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)
        # 使用 build_conv_layer 构建第一个卷积层，卷积核大小为 1×1，步幅由 conv1_stride 控制。然后将 norm1 归一化层添加到模块中。
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        # 如果启用了 dcn（可变形卷积），检查是否需要在步幅大于 1 时使用标准卷积（通过 fallback_on_stride 标志控制）。
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        # 如果没有启用 DCN 或者 fallback_on_stride 为 True，则构建标准卷积层；否则构建可变形卷积层。
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                # 这里的填充与膨胀率相同，用于保持卷积输出的尺寸。
                # 膨胀卷积（dilated convolution）会在卷积核的元素之间插入空隙，从而扩大感受野。
                padding=dilation,
                # 膨胀率，用于控制卷积的感受野大小。dilation=1 表示普通卷积，dilation > 1 则是膨胀卷积。
                dilation=dilation,
                bias=False)
        # 如果启用了 DCN 且不需要回退到标准卷积层，程序会进入这个分支，构建一个可变形卷积层。
        else:
            # assert self.conv_cfg is None: 确保 conv_cfg 必须为 None，因为在使用 DCN 时，不应该再指定标准卷积的配置。
            # 可变形卷积已经是一种特殊的卷积操作，和普通卷积不兼容。
            # 如果这个断言失败（self.conv_cfg 不为 None），则会抛出错误 'conv_cfg must be None for DCN'，
            # 提醒开发者不能同时设置两种卷积配置。
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        # 构建第三个卷积层，卷积核大小为 1×1，输出通道数为输入通道数的 4 倍。并将 norm2 和 norm3 归一化层添加到模块中。
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        # 定义 ReLU 激活函数，inplace=True 表示在原地执行以减少内存开销。
        self.relu = nn.ReLU(inplace=True)
        # 保存 downsample 下采样模块（如果需要）。
        self.downsample = downsample

        # 如果有插件，调用 make_block_plugins 方法为 conv1、conv2 和 conv3 层生成相应的插件。
        # make_block_plugins 和 forward_plugin
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.
        # 定义一个名为 make_block_plugins 的方法，接受两个参数：
        # in_channels：表示传入插件的输入通道数。
        # plugins：一个包含插件配置信息的列表，每个配置是一个字典。
        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        # 使用断言确保 plugins 参数是一个列表。如果不是，程序将在运行时抛出异常，提示错误。
        assert isinstance(plugins, list)
        # 初始化一个空列表 plugin_names，用于存储生成的插件名称。
        plugin_names = []

        for plugin in plugins:  # 遍历 plugins 列表中的每个插件配置字典。
            plugin = plugin.copy()  # 创建插件配置字典的副本，以防止修改原始配置。
            # 调用 build_plugin_layer 函数，生成插件的层（layer）和名称（name）。
            name, layer = build_plugin_layer(
                plugin, # plugin：插件的配置字典。
                in_channels=in_channels, # in_channels：传入插件的输入通道数。
                # 从插件配置中移除 postfix 键，并将其值作为后缀传递给 build_plugin_layer。如果没有 postfix，则使用空字符串作为默认值。
                postfix=plugin.pop('postfix', ''))
            # 断言确保当前模块没有具有相同名称的插件。如果已有相同名称的插件，抛出异常，防止名称冲突。
            assert not hasattr(self, name), f'duplicate plugin {name}'
            # 将生成的插件层（layer）添加到当前模块中，使用 name 作为名称。这样可以确保插件在 Bottleneck 模块中可被访问。
            self.add_module(name, layer)
            # 将插件的名称添加到 plugin_names 列表中，以便在方法结束时返回。
            plugin_names.append(name)
        # 返回 plugin_names 列表，包含所有生成插件的名称。这些名称可用于在前向传播时调用对应的插件层。
        return plugin_names

    # 定义了 forward_plugin 方法，接受两个参数：
    # 输入张量，通常是前一个层的输出
    # plugin_names：一个包含插件名称的列表，这些插件将会在前向传播中被调用。
    def forward_plugin(self, x, plugin_names):
        out = x   # 初始化一个变量 out，将其设置为输入 x。此后，out 将存储经过插件处理后的输出。
        # 这行代码开始了一个循环，将遍历 plugin_names 列表中的每个插件名称。每次循环，name 将对应列表中的一个插件。
        for name in plugin_names:
            # getattr(self, name)：根据 name 从 self 中提取对应的插件方法。
            # (x)：将输入 x 作为参数传入插件方法。
            # 结果将赋值给 out，也就是说，out 的值会被更新为插件处理后的结果。这一过程让每个插件都可以基于前一个插件的输出进行操作。
            out = getattr(self, name)(x)
        # 最后，函数将返回处理完成的 out。这意味着经过所有插件的处理后，out 现在包含了最终的输出结果。
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        # 定义了一个内部函数 _inner_forward，用于执行主要的前向传播计算。
        # identity 是跳跃连接（skip connection）的输入，它保存了原始输入数据，以便稍后加到输出中。
        def _inner_forward(x):
            identity = x
            # 通过第一层卷积层 conv1 对输入 x 进行卷积操作。
            # 然后通过 norm1（归一化层，例如 BatchNorm）进行归一化。
            # 最后通过 relu 激活函数添加非线性变换。
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            # 如果 with_plugins 为真，则在第一层卷积之后调用插件（可选的额外操作，例如特定层的修改或增强）。
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone.
    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        网络前面stem部分的通道数。如果不指定，则与 base_channels 相同。stem 是网络接收输入后最初几层的卷积操作。
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()1
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)

    """
    # ResNet-18: (2, 2, 2, 2)：每个阶段包含 2 个 BasicBlock。
    # ResNet-34: (3, 4, 6, 3)：每个阶段包含 3、4、6 和 3 个 BasicBlock。
    # ResNet-50: (3, 4, 6, 3)：每个阶段包含 3、4、6 和 3 个 Bottleneck。
    # ResNet-101: (3, 4, 23, 3)：每个阶段包含 3、4、23 和 3 个 Bottleneck。
    # ResNet-152: (3, 8, 36, 3)：每个阶段包含 3、8、36 和 3 个 Bottleneck。
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth, # 指定 ResNet 的深度，如 18、34、50、101 或 152
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ResNet, self).__init__(init_cfg)
        # 初始化 zero_init_residual 参数，用于决定是否将残差块中的最后一个归一化层初始化为零。
        self.zero_init_residual = zero_init_residual
        # 检查 depth 是否在 arch_settings 中定义。arch_settings 存储了不同深度的 ResNet 架构信息，如果指定的 depth 不合法，则抛出错误。
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        # 检查 num_stages 的取值范围，确保 strides 和 dilations 的长度与 num_stages 相同，out_indices 中的值要小于 num_stages。
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        # 构建 ResNet 层
        # 根据 depth 设置残差块类型和各阶段的块数量，并调用 _make_stem_layer 方法构建输入的 stem 层。
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        #  构建每个残差层
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)
        #########################################
        ####################################
    # 该函数接受两个参数：
    # plugins：插件的配置列表。每个插件是一个字典，定义了要插入的模块类型及其相关配置。
    # stage_idx：阶段的索引，表示要在 ResNet 的哪个阶段插入插件。
    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        # 遍历传入的插件列表 plugins。在循环中，将每个插件字典复制一份，这样可以避免直接修改原始的插件字典，
        # 保证后续操作对插件的原始定义不会产生副作用。
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(  # 第一个
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(  # 第二个
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(  # 第三个
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        # 从1到 self.frozen_stages 的值（包含）遍历，目的是冻结指定的层。
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        # 遍历 self.res_layers 中的每个层，i 为当前索引，layer_name 为当前层的名称。
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
@BACKBONES.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.
    # 继续说明 ResNetV1d 与默认的 ResNet（ResNetV1b）的区别：
    # 输入干预层的7x7卷积被替换为三个3x3卷积，这样可以更好地提取特征。
    # 在下采样块中，增加了一个步幅为2的2x2平均池化层，在该层之前使用，然后将卷积层的步幅改为1。
    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        """
        deep_stem=True 表示启用深干预层，这会导致在模型的输入干预层使用多个小卷积层而不是一个大卷积层。
        avg_down=True 表示在下采样时使用平均池化（avg_pool），这与传统的使用最大池化的方式有所不同，通常用于减少空间维度的同时保持平滑性。
        **kwargs 是可变参数，允许将其他任何参数传递给父类的构造函数，便于扩展。
        """
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
