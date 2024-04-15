import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector_quantize_pytorch as vq
import copy
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY


class  VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta, use_cosine_sim = True , decay=0):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.decay = decay
        self.use_cosine_sim = use_cosine_sim
        self.codebook = vq.vectorquantizer(self.codebook_size, self.emb_dim, beta=self.beta, use_cosine_sim=self.use_cosine_sim)

    def forward(self, x):
        x, loss, stats = self.codebook(x)
        return x, loss, stats


class SeqVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta, use_cosine_sim=True):
        super(SeqVectorQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.use_cosine_sim = use_cosine_sim
        self.codebook = vq.vectorquantizer(self.codebook_size, self.emb_dim, beta=self.beta,
                                           use_cosine_sim=self.use_cosine_sim)

    def forward(self, x):
        batch, channels, length = x.shape
        quantized = torch.zeros_like(x)
        losses = torch.zeros(batch, length)

        # Assume x is [batch, emb_dim, seq_length]
        for i in range(length):
            frame = x[:, :, i:i + 1]  # Take each frame separately
            q_frame, loss, _ = self.codebook(frame)
            quantized[:, :, i:i + 1] = q_frame
            losses[:, i] = loss

        average_loss = losses.mean()  # or however you want to handle the per-frame loss
        return quantized, average_loss, {}



def normalize(num_channels, type='group'):
    """
    返回指定类型的归一化层。

    Args:
        num_channels (int): 通道数或特征数。
        type (str, optional): 归一化类型。支持'group'和'batch'。默认为'group'。

    Returns:
        nn.Module: PyTorch归一化层。
    """
    if type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)
    elif type == 'batch':
        return nn.BatchNorm2d(num_channels)
    else:
        raise ValueError("Unsupported normalization type. Expected 'group' or 'batch', got {}".format(type))


def swish(x):
    return x*torch.sigmoid(x)


class Downsample1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, l = q.shape
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        w_ = torch.bmm(q, k.transpose(1, 2))
        w_ = w_ * (c ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        h_ = torch.bmm(w_, v.permute(0, 2, 1)).permute(0, 2, 1)
        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, seq_len, attn_resolutions):
        super(Encoder, self).__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.seq_len = seq_len
        self.attn_resolutions = attn_resolutions

        curr_len = self.seq_len
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # 初始卷积层
        blocks.append(nn.Conv1d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # 残差和下采样块
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

            if i != self.num_resolutions - 1:
                blocks.append(Downsample1D(block_in_ch))  # 定义一维的下采样操作
                curr_len = curr_len // 2  # 假设每次下采样都将序列长度减半

        # 注意力块，适用于一维操作
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))  # 定义一维的注意力机制
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # 归一化并转换到潜在大小 - 保持不变
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv1d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @ARCH_REGISTRY.register()
    class VQAutoEncoder(nn.Module):
        def __init__(self, in_channels, nf, ch_mult, res_blocks=2, attn_resolutions=[16], codebook_size=1024,
                     emb_dim=256, beta=0.25, model_path=None):
            super().__init__()
            self.in_channels = in_channels  # 根据输入数据调整，例如LFCC特征的通道数
            self.nf = nf
            self.n_blocks = res_blocks
            self.codebook_size = codebook_size
            self.embed_dim = emb_dim
            self.ch_mult = ch_mult
            self.seq_len = None  # 音频长度
            self.attn_resolutions = attn_resolutions
            self.quantizer_type = "nearest"  # 直接指定为"nearest"
            self.beta = beta


            self.encoder = Encoder(
                self.in_channels,
                self.nf,
                self.embed_dim,
                self.ch_mult,
                self.n_blocks,
                self.seq_len,  # 需要传入序列长度
                self.attn_resolutions
            )


            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)



            # 加载预训练模型的部分可以保留，以支持从预训练模型初始化
            if model_path is not None:
                chkpt = torch.load(model_path, map_location='cpu')
                if 'params_ema' in chkpt:
                    self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'])
                elif 'params' in chkpt:
                    self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
                else:
                    raise ValueError(f'Wrong params!')

        def forward(self, x):
            x = self.encoder(x)
            quant, codebook_loss, quant_stats = self.quantize(x)
            return quant, codebook_loss, quant_stats





