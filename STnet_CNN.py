import torch
import torch.nn as nn
import copy
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from torchvision import models
from functools import partial
import torch.nn.functional as F
from mamba_ssm import Mamba


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, in_kernel, in_pad, in_bias):
#         super(ResBlock, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
#         self.relu1 = nonlinearity
#         self.conv2 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
#         self.relu2 = nonlinearity
#
#     def forward(self, x):
#         x0 = self.conv1(x)
#         x = self.relu2(x0)
#         x = self.conv2(x)
#         x = x0 + x
#         out = self.relu2(x)
#         return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GroupGLKA(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)

        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)

        x = self.proj_last(x * a) * self.scale + shortcut

        return x


class SGAB(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor=15, attn='GLKA'):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()

        self.LKA = GroupGLKA(n_feats)

        self.LFE = SGAB(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x



class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation funtion
    normalization includes BN and IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 groups=1, dilation=1, bias=False, norm=nn.InstanceNorm2d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):

        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out



# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, in_kernel, in_pad, in_bias, act=nn.GELU):
#         super(BasicBlock, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
#         self.relu1 = act()
#         self.conv2 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
#         self.relu2 = act()
#
#     def forward(self, x):
#         x0 = self.conv1(x)
#         x = self.relu2(x0)
#         x = self.conv2(x)
#         x = x0 + x
#         out = self.relu2(x)
#         return out

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.InstanceNorm2d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        # self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)
        # self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1, norm=norm, act=act, preact=preact)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.relu1 = act()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.relu2 = act()
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_ch != out_ch:
        #     self.shortcut = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        x0 = self.conv1(x)
        x = self.relu2(x0)
        x = self.conv2(x)
        x = x0 + x
        out = self.relu2(x)
        return out


class Feature_Extraction(nn.Module):
    def __init__(self, input_nc=1, output_nc=32, stride=1, norm=nn.InstanceNorm2d, act=nn.ReLU, preact=False):
        super(Feature_Extraction, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.act = act()
        if self.input_nc == 2:
            self.norm = norm(self.input_nc)
        self.conv1 = nn.Conv2d(self.input_nc, self.output_nc, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.output_nc, self.output_nc, 3, 1, 1)
        # self.conv1 = BasicBlock(self.input_nc, self.output_nc, stride, norm, act)
        # self.conv2 = BasicBlock(self.output_nc, self.output_nc, stride, norm, act, preact)
        # self.mamba = ResMambaBlock(in_channels=output_nc, out_chanels=output_nc)#参数类型要改统一
        self.conMAB1 = MAB(n_feats=output_nc)
        # self.conMAB2 = MAB(n_feats=output_nc)


    def forward(self, x):

        if self.input_nc == 2:
            x = self.norm(x)
        x = self.conv1(x)
        res = x
        out = self.conMAB1(x)
        out = out + res
        out = self.conv2(out)
        # shortcut = x.clone()
        # out = self.conMAB1(x)
        # out = self.conMAB2(out)
        # out = self.act(out)
        # out = self.conv2(out)
        # out = shortcut + out
        return out


class Feature_Fusion(nn.Module):
    def __init__(self, input_nc=32, output_nc=128, stride=1, norm=nn.InstanceNorm2d, act=nn.ReLU, preact=False):
        super(Feature_Fusion, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.act = act()
        self.norm = norm(self.input_nc)
        # self.feat = nn.Sequential(BasicBlock(self.input_nc, self.input_nc, stride, norm, act, preact),
        #                           MAB(n_feats=input_nc),
        #                           BasicBlock(self.input_nc, self.input_nc, stride, norm, act, preact)
        #                           )
        self.feat = MAB(n_feats=input_nc)
        self.f_feat = MAB(n_feats=input_nc)
        self.b_feat = MAB(n_feats=input_nc)
        self.conv1 = nn.Conv2d(self.input_nc, self.input_nc, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.input_nc, self.output_nc, 3, 1, 1)


    def forward(self, x, f_flow, b_flow):
        x = self.norm(x)
        f_flow = self.norm(f_flow)
        b_flow = self.norm(b_flow)
        shorcut = x
        x = self.feat(x)
        f_flow = self.f_feat(f_flow)
        b_flow = self.b_feat(b_flow)
        f_fusion = x*f_flow
        b_fusion = x*b_flow
        fusion = f_fusion + b_fusion
        fusion = self.conv1(fusion)
        out = fusion + shorcut
        out = self.conv2(out)
        return out


class LKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        # self.norm = LayerNorm(n_feats, data_format='channels_first')
        # self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())

        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 3, groups=n_feats, dilation=3),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats, res_scale=1.0):
        super(ResGroup, self).__init__()
        self.body = nn.ModuleList([
            MAB(n_feats)\
            for _ in range(n_resblocks)])

        self.body_t = LKAT(n_feats)

    def forward(self, x):
        res = x.clone()

        for i, block in enumerate(self.body):
            res = block(res)

        x = self.body_t(res) + x

        return x


class STCnet(nn.Module):
    def __init__(self, input_nc=1, flow_nc=2, output_nc=1, n_feats=32, n_resgroups=4, stride=1, norm=nn.InstanceNorm2d, act=nn.GELU(), preact=False):
        super(STCnet, self).__init__()
        self.input_nc = input_nc
        self.flow_nc = flow_nc
        self.output_nc = output_nc
        self.n_feats = n_feats

        self.norm1 = norm(input_nc)
        self.norm2 = norm(flow_nc)
        self.norm3 = norm(flow_nc)
        self.feat = Feature_Extraction(self.input_nc, self.n_feats, stride, norm, act, preact)
        self.conv = nn.Conv2d(self.n_feats, self.n_feats*4, 1)
        self.f_feat = Feature_Extraction(self.flow_nc, self.n_feats, stride, norm, act, preact)
        self.b_feat = Feature_Extraction(self.flow_nc, self.n_feats, stride, norm, act, preact)

        self.feat_fusion = Feature_Fusion(input_nc=self.n_feats, output_nc=self.n_feats*4, stride=1, norm=norm, act=act, preact=False)


        self.feats = Feature_Extraction(self.n_feats, self.n_feats*4, stride, norm, act, preact)#danshuru shiyan

        # self.body = nn.ModuleList([
        #     ResMambaBlock(in_channels=self.n_feats*4, out_chanels=self.n_feats*4)
        #     for i in range(n_resgroups)])

        # self.head = nn.Conv2d(self.input_nc, self.n_feats*4, 3, 1, 1)

        self.body = nn.ModuleList([
            ResGroup(n_resblocks=4, n_feats=self.n_feats*4)
            for _ in range(n_resgroups)])

        self.body_t = nn.Conv2d(self.n_feats*4, self.n_feats*4, 3, 1, 1)
        self.tail = nn.Conv2d(self.n_feats*4, self.output_nc, 3, 1, 1)

        # self.body_t = BasicBlock(self.n_feats*4, self.n_feats*4, stride=1, norm=norm, act=act, preact=False)
        # self.tail = BasicBlock(self.n_feats*4, self.output_nc, stride=1, norm=norm, act=act, preact=False)
        # self.output = F.tanh

    def forward(self, x, forward_flow, back_flow):

        # xx = self.feat(self.norm1(x))
        x = self.feat(x)
        f = self.f_feat(forward_flow)
        b = self.b_feat(back_flow)
        res = x
        # f = self.f_feat(self.norm2(forward_flow))
        # b = self.b_feat(self.norm3(back_flow))
        #
        xx = self.feat_fusion(x, f, b)

        # xx = self.feats(x) #danshuru shiyan

        for i in self.body:
            xx = i(xx)
        res = self.conv(res)
        out = self.body_t(xx) + res
        out = self.tail(out)
        out = torch.tanh(out)
        # x = self.head(x)
        # res = x
        # for i in self.body:
        #     res = i(res)
        # res = self.body_t(res) + x
        # out = self.tail(res)
        return out



