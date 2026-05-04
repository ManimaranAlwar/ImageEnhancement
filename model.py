import torch
import torch.nn as nn


# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.pool(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w

class TinyTransformerBlock(nn.Module):
    """Lightweight transformer over downsampled spatial tokens.
    - Pools to reduce tokens, applies MHSA + MLP, upsamples back, residual to input size.
    """
    def __init__(self, channels: int, pooled_hw: int = 32, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.pooled_hw = pooled_hw
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_hw, pooled_hw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_p = self.pool(x)                 # (B, C, Ph, Pw)
        Ph = Pw = self.pooled_hw
        tokens = x_p.flatten(2).transpose(1, 2)  # (B, Ph*Pw, C)
        y = self.norm1(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + y                 # Residual 1
        y2 = self.mlp(self.norm2(tokens))
        tokens = tokens + y2                # Residual 2
        x_p2 = tokens.transpose(1, 2).reshape(B, C, Ph, Pw)
        x_up = torch.nn.functional.interpolate(x_p2, size=(H, W), mode='bilinear', align_corners=False)
        return x + x_up                     # Residual back to original

class CNNTransformerModel(nn.Module):
    def __init__(self):
        super(CNNTransformerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.se2 = SEBlock(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.se3 = SEBlock(64)
        # Tiny transformer operating on 64-ch features
        self.transformer = TinyTransformerBlock(channels=64, pooled_hw=32, num_heads=4, mlp_ratio=2.0)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 16, kernel_size=1)
        self.relu6 = nn.ReLU()
        self.conv_out = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x2 = self.se2(x2)
        x3 = self.relu3(self.conv3(x2))
        x3 = self.se3(x3)
        # Transformer on mid-level features
        x3 = self.transformer(x3)
        x4 = self.relu4(self.conv4(x3))
        x5 = self.relu5(self.conv5(x4))
        x6 = self.relu6(self.conv6(x5))
        out = self.conv_out(x6)
        # Residual skip connection from input
        out = out + inp
        out = self.sigmoid(out)
        return out
