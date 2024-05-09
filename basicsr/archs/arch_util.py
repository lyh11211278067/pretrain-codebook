import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger

# 它接受一个模块列表（或一个单独的模块），并初始化这些模块的权重。主要是用于深度学习的神经网络模型初始化。
@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """
    初始化网络权重。

    参数:
    (nn module_list(列表。Module] | n.Module):要初始化的模块。
    scale (float):缩放初始化权重，特别是残差
    块。默认值:1。
    bias_fill (float):要填充偏置的值。默认值:0
    kwargs (dict):初始化函数的其他参数。
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


# 创建由多个相同的基本块（basic_block）组成的神经网络层。
def make_layer(basic_block, num_basic_block, **kwarg):
    """
    通过堆叠相同的块来制作层。

    参数:
    Basic_block (nn.module): nn。基本块的模块类。
    Num_basic_block (int):块数量。

    返回:
    神经网络。顺序的:以n.顺序的方式堆叠块。
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


# 表示一个没有批量归一化（Batch Normalization，简称BN）的残差块。
# 卷积-ReLU-卷积，然后加上输入，形成残差连接。
class ResidualBlockNoBN(nn.Module):
    """
    没有BN的剩余块。

    它的风格是:
    ——Conv-ReLU-Conv - +
    |________________|

    参数:
    num_feat (int):中间特征的通道号。
    默认值:64。
    res_scale (float):剩余比例尺。默认值:1。
    pytorch_init (bool):如果设置为True，则使用pytorch默认的init，
    否则，使用default_init_weights。默认值:False。
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


# 用于上采样（放大）特征图。
class Upsample(nn.Sequential):
    """
    Upsample模块。

    参数:
    scale (int):比例因子。支持的尺度:2^n和3。
    num_feat (int):中间特征的通道号。
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# 用于根据给定的光流场（flow field）来扭曲图像或特征图。光流通常用于描述图像中像素或特征点的运动模式。
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    用光流扭曲图像或特征图。

    参数:
    x(张量):大小为(n, c, h, w)的张量。
    flow(张量):大小为(n, h, w, 2)的张量，正常值。
    Interp_mode (str): 'nearest'或'bilinear'。默认值:“双线性”。
    Padding_mode (str): ' 0 '或'border'或'reflection'。
    默认值:“0”。
    align_corners (bool):在pytorch 1.3之前，默认值为
    align_corners = True。pytorch 1.3之后，默认值为
    align_corners = False。这里，我们使用True作为默认值。

    返回:
    张量:扭曲的图像或特征映射。
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


# 用于根据指定的尺寸类型（比率或形状）调整光流（flow）的大小。光流通常用于计算机视觉任务中，表示图像中像素或特征点在不同帧之间的运动。
def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """
    根据比率或形状调整流的大小。

    参数:
    flow(张量):预先计算的流量。形状[N, 2, H, W]。
    Size_type (str): 'ratio'或'shape'。
    Sizes (list[int | float]):调整大小或最终输出的比率
    形状。
    1)比值顺序应为[ratio_h, ratio_w]。为
    下采样时，比值应小于1.0(即比值)
    < 1.0)。对于上采样，比值应大于1.0(即
    比值> 1.0)。
    2) output_size的顺序应为[out_h, out_w]。
    interp_mode (str):调整大小的插值模式。
    默认值:“双线性”。
    align_corners (bool):是否对齐角。默认值:False。

    返回:
    张量:调整流量大小。
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# 该函数用于执行像素级反洗牌（unshuffle）操作的。这种操作通常用于将经过像素洗牌（shuffle）操作后的特征图恢复回原始的形状。
# 像素洗牌通常用于将特征图的通道数减少，同时将空间分辨率提高，这在某些神经网络结构（如PixelShuffle）中用于上采样。
# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """
    像素unshuffle。

    参数:
    x(张量):形状为(b, c, hh, hw)的输入特征。
    scale (int):下采样比。

    返回:
    张量:像素未洗牌特征。
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# 这个类实现了一个可变形对齐的调制可变形卷积操作，它不同于官方版本的DCNv2Pack，具有一些特殊的功能，如生成偏移和蒙版。
class DCNv2Pack(ModulatedDeformConvPack):
    """
    用于可变形对齐的调制可变形转换。

    不同于官方的DCNv2Pack，它生成偏移和蒙版
    与前面的特性不同，这个DCNv2Pack有另一个不同之处
    生成偏移和蒙版的功能。

    裁判:
    深入研究视频超分辨率的可变形对齐。
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)

# 实现了一个截断正态分布（truncated normal distribution）的初始化方法，用于初始化PyTorch中的张量（tensor）。
# 截断正态分布是在给定区间 [a, b] 内对正态分布进行截断得到的分布。
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # 值是通过使用截断均匀分布和
        # 然后对正态分布使用逆CDF。
        # 获取cdf的上下值
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # 用[low, up]的值均匀填充张量，然后转换为[2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # 对正态分布使用逆cdf变换得到截断标准常态
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


# 包装函数，它调用 _no_grad_trunc_normal_ 函数来初始化一个张量，使其值服从在 [a, b] 区间内截断的正态分布。
# 这个截断正态分布意味着，从标准正态分布中抽取的值如果落在 [a, b] 区间之外，则会被重新抽取，直到它们落在该区间内为止。
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""
    使用从截断的对象中绘制的值填充输入张量正态分布。

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.


    参数:
    张量:一个n维的火炬。张量的
    均值:正态分布的均值
    Std:正态分布的标准差
    A:最小截止值
    B:最大截止值

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
# 工厂函数，它返回一个函数，该函数可以将输入转换为指定长度的元组。
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple