"""Transformer decoder block and stack.

Masked self-attention + MLP + residual connections + LayerNorm, stacked
N times, with positional embeddings over the combined [visual tokens] +
[letter tokens] sequence.
"""


import torch
import torch.nn as nn

from src.model.attention import MultiHeadSelfAttention


class MLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, expansion=4, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return x


def build_mask(n_visual: int, n_letters: int, device=None) -> torch.Tensor:
    """
    Masque (T,T), T = n_visual + n_letters, bool. True = attention autorisee.
    - Visuel <-> visuel : tout autorise (bidirectionnel)
    - Lettres -> visuel : autorise (les lettres ont besoin de l'image)
    - Visuel -> lettres : JAMAIS autorise
    - Lettres <-> lettres : causal (chaque lettre voit les precedentes + elle-meme)
    """
    T = n_visual + n_letters
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)

    mask[:n_visual, :n_visual] = True   # visuel -> visuel

    mask[n_visual:, :n_visual] = True   # lettres -> visuel

    causal = torch.tril(torch.ones(n_letters, n_letters, dtype=torch.bool, device=device))
    mask[n_visual:, n_visual:] = causal  # lettres -> lettres, causal

    # visuel -> lettres reste False (deja initialise a zeros)
    return mask


if __name__ == "__main__":
    torch.manual_seed(0)
    n_visual, n_letters = 4, 3
    mask = build_mask(n_visual, n_letters)
    print(mask.int())
    decoder = TransformerDecoder(d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(2, n_visual + n_letters, 32)
    out = decoder(x, mask)
    print("output shape:", out.shape)
    assert out.shape == x.shape
    print("OK")
