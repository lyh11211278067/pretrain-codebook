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




