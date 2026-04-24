from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .video import read_video, write_video


PROXY_NAMES = ("internvl3", "qwen3vl", "videomae")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF = (0.5, 0.5, 0.5)


@dataclass
class MIFGSMConfig:
    epsilon: float = 16 / 255
    alpha: float = 2 / 255
    steps: int = 10
    mu: float = 1.0


class Proxy:
    def __init__(self, model, extractor):
        self.model = model
        self.extractor = extractor

    def forward(self, x):
        return self.extractor(self.model, x)


def _normalize(x, mean, std):
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


def _resize(x, size, mode="bilinear"):
    return F.interpolate(x, size=(size, size), mode=mode, align_corners=False, antialias=True)


def _internvl_extract(model, x):
    x = _resize(x, 448)
    x = _normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return model.vision_model(x).last_hidden_state.mean(1)


def _videomae_extract(model, x):
    T = x.shape[0]
    if T != 16:
        idx = torch.linspace(0, T - 1, 16, device=x.device).long()
        x = x.index_select(0, idx)
    x = _resize(x, 224)
    x = _normalize(x, HALF, HALF)
    return model(pixel_values=x.unsqueeze(0)).last_hidden_state.mean(1)


def _qwen3vl_extract(model, x):
    visual = model.visual
    cfg = visual.config
    Ps, Tp, Ms = cfg.patch_size, cfg.temporal_patch_size, cfg.spatial_merge_size

    x = _resize(x, 448, mode="bicubic").clamp(0, 1)
    x = _normalize(x, HALF, HALF)

    T = x.shape[0]
    if T % Tp != 0:
        pad = x[-1:].expand(Tp - (T % Tp), -1, -1, -1)
        x = torch.cat([x, pad], dim=0)
        T = x.shape[0]

    grid_t = T // Tp
    grid_h = grid_w = 448 // Ps

    p = x.view(1, grid_t, Tp, 3,
               grid_h // Ms, Ms, Ps,
               grid_w // Ms, Ms, Ps)
    p = p.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
    pixel_values = p.reshape(grid_t * grid_h * grid_w, 3 * Tp * Ps * Ps)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=x.device)
    return visual(pixel_values, grid_thw=grid_thw).last_hidden_state


def mi_fgsm(src, dst, proxy, cfg=None, device="cuda"):
    cfg = cfg or MIFGSMConfig()
    frames, fps = read_video(src)
    x0 = torch.from_numpy(frames).to(device).float() / 255.0
    x0 = x0.permute(0, 3, 1, 2).contiguous()

    with torch.no_grad():
        clean = proxy.forward(x0).detach()

    x = x0.clone()
    g = torch.zeros_like(x)
    cos = torch.nn.functional.cosine_similarity
    for _ in range(cfg.steps):
        x.requires_grad_(True)
        loss = 1 - cos(proxy.forward(x), clean, dim=-1).mean()
        grad, = torch.autograd.grad(loss, x)
        g = cfg.mu * g + grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        with torch.no_grad():
            x = (x + cfg.alpha * g.sign()).clamp(x0 - cfg.epsilon, x0 + cfg.epsilon).clamp(0, 1)
        x = x.detach()

    adv = (x.clamp(0, 1) * 255).byte().cpu().numpy()
    write_video(np.transpose(adv, (0, 2, 3, 1)), dst, fps=fps)


def load_proxy(name, device="cuda"):
    from transformers import AutoModel, VideoMAEModel

    if name == "internvl3":
        m = AutoModel.from_pretrained(
            "OpenGVLab/InternVL3-8B", torch_dtype=torch.float32, trust_remote_code=True,
        ).to(device).eval()
        extractor = _internvl_extract
    elif name == "qwen3vl":
        m = AutoModel.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device).eval()
        extractor = _qwen3vl_extract
    elif name == "videomae":
        m = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-large", torch_dtype=torch.float32,
        ).to(device).eval()
        extractor = _videomae_extract
    else:
        raise ValueError(f"unknown proxy: {name}")

    for p in m.parameters():
        p.requires_grad_(False)
    return Proxy(m, extractor)
