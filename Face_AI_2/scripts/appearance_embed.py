"""Frozen face embeddings of the 24 high-resolution crops per video (data/cache_v2/appear/*.npy, 256x256, 2x the face
box), for the v2 appearance routes (age, BMI, sex).

For every encoder and crop width it saves features/v2_<name>.npz with
    keys      (1200,)            video keys in df order
    features  (1200, D)          mean over the 24 crops (what the ridge/PLS/kernel candidates use)
    crops     (1200, 24, D) f16  one vector per crop (for the networks that learn from single crops)

    .venv/Scripts/python.exe scripts/appearance_embed.py <encoder> [<encoder> ...]
Encoders: dinov2_s, dinov2_b, dinov2_l, clip_b16, clip_l14, clip_l14_336, siglip2_b16, siglip2_so400m
Crop widths: "wide" = the whole 256 px crop (2.0x the face box), "tight" = the central 180 px (about 1.4x).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_core as C                                                        # noqa: E402

ENCODERS = {
    "dinov2_s": ("facebook/dinov2-small", "dino"), "dinov2_b": ("facebook/dinov2-base", "dino"),
    "dinov2_l": ("facebook/dinov2-large", "dino"), "clip_b16": ("openai/clip-vit-base-patch16", "clip"),
    "clip_l14": ("openai/clip-vit-large-patch14", "clip"), "clip_l14_336": ("openai/clip-vit-large-patch14-336", "clip"),
    "siglip2_b16": ("google/siglip2-base-patch16-224", "siglip"), "siglip2_so400m": ("google/siglip2-so400m-patch14-384", "siglip"),
}
ENCODERS["mefem_b"] = ("weights/face_downloads/mefem/MeFEm-B.pth.tar", "timm_vit_b16")   # CC BY 4.0, loose 2x face crops
CROPS = {"wide": slice(0, 256), "tight": slice(38, 218)}


class TimmViT:
    """A timm ViT-B/16 with local weights (MeFEm), returning the same outputs as the transformers models."""
    def __init__(self, path):
        import timm
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="")
        self.model.load_state_dict(torch.load(C.PROJECT / path, map_location="cpu", weights_only=True), strict=True)
        self.model = self.model.eval().cuda().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def __call__(self, imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).cuda().permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        tok = self.model.forward_features(((x - self.mean) / self.std).half())
        return torch.cat([tok[:, 0], tok[:, 1:].mean(1)], 1)


def load_encoder(hf_id, kind):
    if kind == "timm_vit_b16":
        return None, TimmViT(hf_id)
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection
    proc = AutoImageProcessor.from_pretrained(hf_id)
    if kind == "clip":
        model = CLIPVisionModelWithProjection.from_pretrained(hf_id, torch_dtype=torch.float16)
    elif kind == "siglip":
        model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float16).vision_model
    else:
        model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float16)
    return proc, model.eval().cuda()


@torch.inference_mode()
def embed(name, crop):
    hf_id, kind = ENCODERS[name]
    proc, model = load_encoder(hf_id, kind)
    keys = C.df["key"].to_numpy()
    feats = None
    t0 = time.time()
    for i, key in enumerate(keys):
        imgs = np.load(C.CACHE2 / "appear" / f"{key}.npy")[:, CROPS[crop], CROPS[crop]]       # (24, h, w, 3) RGB
        if kind == "timm_vit_b16":
            vec = model(imgs).float().cpu().numpy()
            if feats is None:
                feats = np.zeros((len(keys), imgs.shape[0], vec.shape[1]), np.float16)
            feats[i] = vec
            continue
        px = proc(images=list(imgs), return_tensors="pt")["pixel_values"].cuda().half()
        out = model(pixel_values=px)
        if kind == "clip":
            vec = torch.cat([out.image_embeds, out.last_hidden_state[:, 0]], 1)               # projection + CLS
        elif kind == "siglip":
            vec = torch.cat([out.pooler_output, out.last_hidden_state.mean(1)], 1)            # attention pool + mean
        else:
            vec = torch.cat([out.last_hidden_state[:, 0], out.last_hidden_state[:, 1:].mean(1)], 1)  # CLS + mean patch
        vec = vec.float().cpu().numpy()
        if feats is None:
            feats = np.zeros((len(keys), imgs.shape[0], vec.shape[1]), np.float16)
        feats[i] = vec
        if i % 200 == 0:
            print(f"  {name}/{crop}: {i}/{len(keys)} videos {time.time() - t0:.0f}s", flush=True)
    path = C.FEATURES / f"v2_{name}_{crop}.npz"
    np.savez_compressed(path, keys=keys.astype(str), features=feats.astype(np.float32).mean(1), crops=feats)
    print(f"saved {path.name} {feats.shape} in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    names = sys.argv[1:] or ["dinov2_s", "clip_b16"]
    for n in names:
        for crop in CROPS:
            if not (C.FEATURES / f"v2_{n}_{crop}.npz").exists():
                embed(n, crop)
