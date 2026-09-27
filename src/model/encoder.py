"""CNN vision encoder.

Built from nn.Conv2d, activations, normalization and pooling (no
pretrained backbones). Turns a (B, 3, 64, 64) image into a feature map
(B, C, H', W').

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as encoder(images).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # kernel=3, stride=2, padding=1 : divise H et W exactement par 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.block1 = ConvBlock(3, 32)         # (B,3,64,64)  -> (B,32,32,32)
        self.block2 = ConvBlock(32, 64)        # (B,32,32,32) -> (B,64,16,16)
        self.block3 = ConvBlock(64, d_model)   # (B,64,16,16) -> (B,d_model,8,8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


if __name__ == "__main__":
    enc = CNNEncoder(d_model=128)
    dummy = torch.randn(4, 3, 64, 64)
    out = enc(dummy)
    print("output shape:", out.shape)  # attendu: (4, 128, 8, 8)
    assert out.shape == (4, 128, 8, 8)
    print("OK")
