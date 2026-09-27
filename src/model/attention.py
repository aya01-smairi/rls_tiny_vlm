"""Handwritten multi-head self-attention.

Q/K/V projections, split heads, scaled dot-product attention, mask,
softmax, merge heads, output projection -- written with tensor
operations, not nn.MultiheadAttention. See tests/test_attention.py for
the required equivalence test against F.scaled_dot_product_attention.

The provided tests expect the last nn.Module defined in this file to be
Cls(d_model, n_heads), called as attention(x, mask) with x of shape
(B, T, d_model). `mask` is optional; when given it is a boolean tensor where
True means "this query may attend to this key".
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit etre divisible par n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape

        # 1) Q, K, V en un seul coup : (B, T, 3*d_model)
        qkv = self.qkv_proj(x)

        # 2) Separer Q, K, V : chacun (B, T, d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # 3) Split en tetes : (B, T, d_model) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 4) Scores : Q @ K^T / sqrt(d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)

        # 5) Masque : -inf la ou l'attention est interdite
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # 6) Softmax : distribution de poids qui somme a 1
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7) Ponderer V par ces poids
        out = attn_weights @ v  # (B, H, T, d_head)

        # 8) Fusionner les tetes
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 9) Projection de sortie
        out = self.out_proj(out)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    mask = torch.ones(10, 10, dtype=torch.bool)
    out = attn(x, mask)
    print("output shape:", out.shape)
    assert out.shape == x.shape
    print("OK")
