# Copyright (c) OpenMMLab. All rights reserved.
# Our work is inspired by and builds upon prior methods such as UDMDet and GCCNet.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS




class SF(nn.Module):
    def __init__(self, channel):
        super(SF, self).__init__()
        self.conv1 = Conv(256, channel, 1)
        self.conv2 = Conv(256, channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p4_2 = self.conv1(p4)

        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')

        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
class MultiHeadChannelAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.fc_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 生成 Query
        self.fc_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 生成 Key
        self.fc_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 生成 Value
        self.fc_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 合并输出

    def forward(self, x):
        B, C, H, W = x.size()
        # 生成 Q、K 和 V
        q = self.fc_q(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.fc_k(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.fc_v(x).view(B, self.num_heads, self.head_dim, H * W)

        # 计算注意力
        attn_weights = torch.einsum("bhdw,bhdz->bhwz", q, k)  # (B, num_heads, H*W, H*W)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        out = torch.einsum("bhwz,bhdz->bhdw", attn_weights, v)  # (B, num_heads, head_dim, H*W)
        # out = out.view(B, C, H, W)
        out = out.reshape(B, C, H, W)

        return self.fc_out(out) + x  # 残差连接

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CA_Block(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super(CA_Block, self).__init__()
        self.channel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.multihead_attn = MultiHeadChannelAttention(in_dim, num_heads)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        # 计算通道注意力
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = nn.functional.softmax(energy, dim=-1)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        # 通过多头通道注意力
        out = self.multihead_attn(out)

        return out

class SA_Block(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super(SA_Block, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.multihead_attn = MultiHeadChannelAttention(in_dim, num_heads)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        # 通过多头通道注意力
        out = self.multihead_attn(out)

        return out

class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.GroupNorm(num_groups=32, num_channels=self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 3, 1, 1),
                                    nn.GroupNorm(num_groups=32, num_channels=self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce
###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(x)

        FM = cab +sab
        return FM
###################################################################
# ########################  Interference Mining Feature Module ####
###################################################################
class IFM(nn.Module):
    def __init__(self, channel1, channel2):
        super(IFM, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        # self.mask = nn.Sequential(nn.Conv2d(self.channel1, 1, 7, 1, 3), nn.ReLU())
        self.high_feature = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 3, 1, 1),
                                          nn.GroupNorm(num_groups=32, num_channels=self.channel1),
                                          nn.ReLU())
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=self.channel1)
        self.relu1 = nn.ReLU(inplace=True)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=self.channel1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # y: higher-level features
        # in_map: higher-level c
        input_map = self.high_feature(y)
        up = F.interpolate(input_map, size=x.size()[2:], mode='bilinear', align_corners=True)
        mask = self.sa(up)
        mask = self.sigmoid(mask)

        f_feature = x * mask
        b_feature = x * (1 - mask)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = (self.beta * fn) + x + up
        refine1 = self.gn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 - (self.alpha * fp)
        refine2 = self.gn2(refine2)
        refine2 = self.relu2(refine2)

        return refine2

class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()

    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
@NECKS.register_module()
class CIDNet(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False):
        super(CIDNet, self).__init__()

        _, C3_size, C4_size, C5_size = in_channels
        feature_size = out_channels
#############################
        self.reduce_c3 = nn.Sequential(
            nn.Conv2d(C3_size, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.reduce_c4 = nn.Sequential(
            nn.Conv2d(C4_size, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.reduce_c5 = nn.Sequential(
            nn.Conv2d(C5_size, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        ###############
        self.P6_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=256),
                                  nn.ReLU())
        self.positioning = Positioning(256)
        self.P6_2 = nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1)

        self.P7_1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(256, feature_size, kernel_size=3, stride=2, padding=1))

        self.P5_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=256),
                                  nn.ReLU())
        self.focus3 = IFM(256, 256)
        self.P5_2 = nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1)


        self.P4_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=256),
                                  nn.ReLU())
        self.focus2 = IFM(256, 256)
        self.P4_2 = nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(num_groups=32, num_channels=256),
                                  nn.ReLU())
        self.focus1 = IFM(256, 256)
        self.P3_2 = nn.Conv2d(256, feature_size, kernel_size=3, stride=1, padding=1)
#####################################################################
        self.ScalSeq = SF(256)
        self.add = Add()  # 实例化 Add 模块
    def forward(self, inputs):
        C2, C3, C4, C5 = inputs    #  3 4  5   512 1024 2048

        P3_pre = self.reduce_c3(C3)
        P4_pre = self.reduce_c4(C4)
        P5_pre = self.reduce_c5(C5)

        P6_x = self.P6_1(P5_pre)
        P6_x = self.positioning(P6_x)

        P5_x = self.P5_1(P5_pre)
        P5_x_x = P5_x
        P5_x = self.focus3(P5_x, P6_x)

        P4_x = self.P4_1(P4_pre)
        P4_x_x = P4_x
        P4_x = self.focus2(P4_x, P5_x)

        P3_x = self.P3_1(P3_pre)
        P3_x_x = P3_x
        P3_x = self.focus1(P3_x, P4_x)
###########################################################################
        multi_scale_features_1 = [P3_x_x, P4_x_x, P5_x_x]
        input2 = self.ScalSeq(multi_scale_features_1)
        P3_x_p = self.add([P3_x, input2])
        P3_x = P3_x_p

        P3_x_p = F.interpolate(P3_x_p, size=(P4_x.size(2), P4_x.size(3)), mode='bilinear', align_corners=False)
        P4_x_p = self.add([P4_x, P3_x_p])
        P4_x = P4_x_p

        P4_x_p = F.interpolate(P4_x_p, size=(P5_x.size(2), P5_x.size(3)), mode='bilinear', align_corners=False)
        P5_x_p = self.add([P5_x, P4_x_p])
        P5_x = P5_x_p

        P5_x_p = F.interpolate(P5_x_p, size=(P6_x.size(2), P6_x.size(3)), mode='bilinear', align_corners=False)
        P6_x_p = self.add([P6_x, P5_x_p])
        P6_x = P6_x_p
######################################################################
        P7_x = self.P7_1(P6_x)
        P6_x = self.P6_2(P6_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_2(P3_x)

        outs = [P3_x, P4_x, P5_x, P6_x, P7_x]
        return tuple(outs)
