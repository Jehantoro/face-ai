"""Camera-pulse network definitions for rppg_deep.py. Every model takes raw frames (B, 3, T, H, W) in 0..255 and
returns one pulse value per frame (B, T); the input normalisation lives inside the wrapper, so training and inference
always match.

build_model(name) -> (model, spec) with spec = {"T": frames per window, "size": frame size}.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "rppg"


# ------------------------------------------------------------------ PhysNet (Yu et al., BMVC 2019)
class PhysNet(nn.Module):
    """PhysNet_padding_Encoder_Decoder_MAX: the layer names follow the original release, so its state dict loads."""
    def __init__(self, frames=160, cin=3):
        super().__init__()
        def blk(cin, cout, k, p):
            return nn.Sequential(nn.Conv3d(cin, cout, k, stride=1, padding=p), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))
        self.ConvBlock1 = blk(cin, 16, [1, 5, 5], [0, 2, 2])
        self.ConvBlock2 = blk(16, 32, [3, 3, 3], 1)
        self.ConvBlock3 = blk(32, 64, [3, 3, 3], 1)
        self.ConvBlock4 = blk(64, 64, [3, 3, 3], 1)
        self.ConvBlock5 = blk(64, 64, [3, 3, 3], 1)
        self.ConvBlock6 = blk(64, 64, [3, 3, 3], 1)
        self.ConvBlock7 = blk(64, 64, [3, 3, 3], 1)
        self.ConvBlock8 = blk(64, 64, [3, 3, 3], 1)
        self.ConvBlock9 = blk(64, 64, [3, 3, 3], 1)
        self.upsample = nn.Sequential(nn.ConvTranspose3d(64, 64, [4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
                                      nn.BatchNorm3d(64), nn.ELU())
        self.upsample2 = nn.Sequential(nn.ConvTranspose3d(64, 64, [4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
                                       nn.BatchNorm3d(64), nn.ELU())
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.frames = frames

    def forward(self, x):
        T = x.shape[2]
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        x = F.adaptive_avg_pool3d(x, (T, 1, 1))
        return self.ConvBlock10(x).view(-1, T)


class Standardize(nn.Module):
    """Per-clip standardisation of raw frames (the 'Standardized' input of the rPPG literature)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = x / 255.0
        m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        s = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6
        return self.net((x - m) / s)


class DiffNormalize(nn.Module):
    """Frame differences normalised by the frame sums ('DiffNormalized'), concatenated with the standardised frames."""
    def __init__(self, net, both=True):
        super().__init__()
        self.net, self.both = net, both

    def forward(self, x):
        x = x / 255.0 + 1e-3
        d = (x[:, :, 1:] - x[:, :, :-1]) / (x[:, :, 1:] + x[:, :, :-1])
        d = torch.cat([d, d[:, :, -1:]], 2)
        d = d / (d.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6)
        if not self.both:
            return self.net(d)
        m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        s = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6
        return self.net(torch.cat([d, (x - m) / s], 1))


def _state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


class RhythmFormerNet(nn.Module):
    """RhythmFormer (Zou et al. 2024; MIT). Input: per-clip standardised RGB frames, 128 px, 160 frames."""
    def __init__(self, weights=None):
        super().__init__()
        from third_party.rhythmformer.RhythmFormer import RhythmFormer
        self.net = RhythmFormer()
        if weights:
            self.net.load_state_dict(_state_dict(weights), strict=True)

    def forward(self, x):                                       # x: (B, 3, T, H, W) in 0..255
        m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        s = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6
        return self.net(((x - m) / s).permute(0, 2, 1, 3, 4).contiguous())


class PhysFormerNet(nn.Module):
    """PhysFormer (Yu et al. CVPR 2022; MIT). Input: (x - 127.5) / 128, 128 px, 160 frames."""
    def __init__(self, weights=None, bgr=False):
        super().__init__()
        from third_party.physformer.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
        self.net = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
                                                    num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
        self.bgr = bgr
        if weights:
            self.net.load_state_dict(_state_dict(weights), strict=True)

    def forward(self, x):
        if self.bgr:
            x = x.flip(1)
        return self.net((x - 127.5) / 128.0, 2.0)[0]


class ContrastPhysNet(nn.Module):
    """Contrast-Phys PhysNet(S=2) (Sun & Li, ECCV 2022; MIT). Normalises inside; we use its spatial-average output."""
    def __init__(self, weights=None):
        super().__init__()
        from third_party.contrastphys.PhysNetModel import PhysNet as CPPhysNet
        self.net = CPPhysNet(S=2)
        if weights:
            self.net.load_state_dict(_state_dict(weights), strict=True)

    def forward(self, x):
        return self.net(x)[:, -1]


def build_model(name):
    if name == "physnet":                          # PhysNet trained from scratch on MCD, standardised frames, 72 px
        return Standardize(PhysNet()), {"T": 160, "size": 72}
    if name == "physnet_diff":                     # PhysNet on frame differences + standardised frames (6 channels)
        return DiffNormalize(PhysNet(cin=6), both=True), {"T": 160, "size": 72, "fp32": True}
    if name.startswith("rhythmformer"):            # rhythmformer_pure / _mmpd / _vipl / _scratch
        src = name.split("_", 1)[1] if "_" in name else "pure"
        w = {"pure": "PURE_cross_RhythmFormer.pth", "mmpd": "MMPD_intra_RhythmFormer.pth",
             "vipl": "VIPL_RhythmFormer_fold1.pth", "scratch": None}[src]
        return RhythmFormerNet(WEIGHTS / w if w else None), {"T": 160, "size": 128}
    if name.startswith("physformer"):              # physformer_vipl / physformer_vipl_bgr / physformer_scratch
        w = None if "scratch" in name else WEIGHTS / "Physformer_VIPL_fold1.pkl"
        return PhysFormerNet(w, bgr=name.endswith("_bgr")), {"T": 160, "size": 128}
    if name.startswith("contrastphys"):            # contrastphys_ubfc / contrastphys_scratch
        w = None if "scratch" in name else WEIGHTS / "contrastphys_ubfc.pt"
        return ContrastPhysNet(w), {"T": 160, "size": 128}
    raise ValueError(f"unknown model {name}")
