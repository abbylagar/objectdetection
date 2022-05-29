"""Attention Module 
The Attention module is the core of the vision transformer model. It implements the attention mechanism:

1) Multiply QKV by their weights 2) Perform dot product on Q and K. 3) Normalize the result in 2) by sqrt of head_dim
4) Softmax is applied to the result. 5) Perform dot product on the result of 4) and V and the result is the output.
"""

import torch
from torch import nn



class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x