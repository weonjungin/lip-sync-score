# lipsyncscore/models/syncnet_like.py
import torch.nn as nn
from .lip_encoder import LipEncoder
from .audio_encoder import AudioEncoder


class SyncNetLike(nn.Module):
    def __init__(self, in_frames=5, emb_dim=256):
        super().__init__()
        self.lip = LipEncoder(in_frames=in_frames, emb_dim=emb_dim)
        self.audio = AudioEncoder(emb_dim=emb_dim)

    def forward_lip(self, lips):
        return self.lip(lips)

    def forward_audio(self, mel):
        return self.audio(mel)
