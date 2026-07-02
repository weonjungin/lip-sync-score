# lipsyncscore/models/audio_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    """
    SyncNet-style audio encoder.
    Input:  (B, 80, T)  -> treated as (B, 1, 80, T)
    Output: (B, D)
    """
    def __init__(self, emb_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        # x: (B, 80, T)
        x = x.unsqueeze(1)  # (B, 1, 80, T)
        x = self.conv(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x
