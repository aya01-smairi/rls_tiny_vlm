"""Full vision-language model.

Wires together the CNN encoder, the adapter (flatten + linear projection
to visual tokens), the Transformer decoder, and the linear head that
produces next-letter logits.

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as model(images, input_ids), with
input_ids of shape (B, T), returning logits of shape (B, T_out, 27) where
T_out >= T and the last T logits line up with input_ids.
"""


import torch
import torch.nn as nn

from src.model.encoder import CNNEncoder
from src.model.decoder import TransformerDecoder, build_mask


class TinyVLM(nn.Module): 
    def __init__(self, vocab_size: int = 27, pad_id: int = 26, d_model: int = 128, 
                 n_heads: int = 4, n_layers: int = 4, max_letters: int = 45, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.cnn = CNNEncoder(d_model=d_model)
        self.n_visual_tokens = 8 * 8  # 64, d'apres la sortie du CNN

        self.visual_proj = nn.Linear(d_model, d_model)
        self.visual_pos_emb = nn.Parameter(torch.randn(1, self.n_visual_tokens, d_model) * 0.02)

        self.letter_emb = nn.Embedding(vocab_size, d_model)
        self.letter_pos_emb = nn.Parameter(torch.randn(1, max_letters, d_model) * 0.02)

        self.decoder = TransformerDecoder(d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(d_model, vocab_size)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(images)                       # (B, d_model, 8, 8)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)         # (B, 64, d_model)
        visual_tokens = self.visual_proj(feat)
        visual_tokens = visual_tokens + self.visual_pos_emb
        return visual_tokens

    def forward(self, images: torch.Tensor, tokens_in: torch.Tensor) -> torch.Tensor:
        B, T_letters = tokens_in.shape

        visual_tokens = self.encode_image(images)

        letter_tokens = self.letter_emb(tokens_in)
        letter_tokens = letter_tokens + self.letter_pos_emb[:, :T_letters, :]

        x = torch.cat([visual_tokens, letter_tokens], dim=1)

        mask = build_mask(self.n_visual_tokens, T_letters, device=x.device)

        x = self.decoder(x, mask)

        # Positions correspondant aux lettres : la sortie a la DERNIERE
        # position visuelle predit deja la PREMIERE lettre (pas de token
        # <start> necessaire).
        letter_positions = x[:, self.n_visual_tokens - 1 : -1, :]

        logits = self.head(letter_positions)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, pad_id = 28, 27
    model = TinyVLM(vocab_size=vocab_size, pad_id=pad_id, d_model=64, n_heads=4, n_layers=2)

    B, T = 2, 10
    images = torch.randn(B, 3, 64, 64)
    tokens_in = torch.randint(0, 26, (B, T))

    print("images:", images.shape)
    visual = model.encode_image(images)
    print("visual tokens:", visual.shape)

    logits = model(images, tokens_in)
    print("logits:", logits.shape)  # attendu: (2, 10, 28)
    assert logits.shape == (B, T, vocab_size)
    print("OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de parametres: {n_params:,}")
