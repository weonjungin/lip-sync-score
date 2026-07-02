# lipsyncscore/models/lip_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LipEncoder(nn.Module):
    """
    SyncNet-style lip encoder.
    Input:  (B, N, H, W)  -> treat N frames as channels
    Output: (B, D)
    """
    def __init__(self, in_frames=5, emb_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_frames, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        # x: (B, N, H, W)
        x = self.conv(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x
