import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .video import read_video, write_video


def gaussian_noise(src, dst, sigma=15.0, seed=42):
    frames, fps = read_video(src)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, frames.shape).astype(np.float32)
    out = np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    write_video(out, dst, fps=fps)


def motion_blur(src, dst, kernel_range=(9, 21), angle_jitter=180.0, seed=42):
    frames, fps = read_video(src)
    rng = np.random.default_rng(seed)
    lo, hi = kernel_range
    k = int(rng.integers(lo // 2, hi // 2 + 1)) * 2 + 1
    angle = float(rng.uniform(-angle_jitter, angle_jitter))

    base = np.zeros((k, k), dtype=np.float32)
    base[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kern = cv2.warpAffine(base, M, (k, k))
    s = kern.sum()
    if s > 0:
        kern /= s

    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        out[i] = cv2.filter2D(f, -1, kern)
    write_video(out, dst, fps=fps)


def compress(src, dst, target_ratio=0.1519, **_):
    src, dst = Path(src), Path(dst)
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=bit_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip().isdigit():
        raise RuntimeError(f"ffprobe failed to read bitrate from {src}: {r.stderr.strip()}")
    src_bitrate = int(r.stdout.strip())
    target = max(64_000, int(src_bitrate * target_ratio))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-b:v", str(target),
         "-maxrate", str(int(target * 1.2)),
         "-bufsize", str(target * 2),
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        check=True,
    )


def _overlay(frame, text, scale=0.9):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 8
    x = (w - tw) // 2
    y = h - 30
    over = frame.copy()
    cv2.rectangle(over, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
    frame = cv2.addWeighted(over, 0.55, frame, 0.45, 0)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def caption_injection(src, dst, misleading=None, irrelevant_pool=None,
                      coverage=0.7, mislead_every=5, font_scale=0.9, seed=42):
    if not irrelevant_pool:
        raise ValueError("caption_injection needs irrelevant_pool")
    if isinstance(misleading, str):
        misleading = [misleading]
    elif misleading is not None:
        misleading = list(misleading)

    frames, fps = read_video(src)
    T = len(frames)
    chunk = max(1, int(round(fps)))
    n_chunks = max(1, T // chunk)
    n_capped = int(round(n_chunks * coverage))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(n_chunks), min(n_capped, n_chunks)))

    out = frames.copy()
    mislead_n = 0
    for i, cid in enumerate(chosen):
        if misleading and i % mislead_every == 0:
            text = misleading[mislead_n % len(misleading)]
            mislead_n += 1
        else:
            text = rng.choice(irrelevant_pool)
        a, b = cid * chunk, min(T, cid * chunk + chunk)
        for t in range(a, b):
            out[t] = _overlay(out[t], text, font_scale)
    write_video(out, dst, fps=fps)


_ALPHA = list("abcdefghijklmnopqrstuvwxyz ")


def _corrupt(text, rng, p_sub, p_del, p_ins):
    o = []
    for c in text:
        r = rng.random()
        if r < p_del:
            continue
        if r < p_del + p_sub and c.isalpha():
            o.append(rng.choice(_ALPHA))
        else:
            o.append(c)
        if rng.random() < p_ins:
            o.append(rng.choice(_ALPHA))
    return "".join(o)


def subtitle_corruption(src, dst, transcript, p_sub=0.10, p_del=0.05, p_ins=0.05,
                        shift_range=(0.5, 2.0), font_scale=0.9, seed=42):
    frames, fps = read_video(src)
    T = len(frames)
    rng = random.Random(seed)
    shift = rng.uniform(*shift_range) * rng.choice([-1.0, 1.0])
    out = frames.copy()
    for start, end, text in transcript or []:
        noisy = _corrupt(text, rng, p_sub, p_del, p_ins)[:80]
        s = max(0, int(round((start + shift) * fps)))
        e = min(T, int(round((end + shift) * fps)))
        for t in range(s, e):
            out[t] = _overlay(out[t], noisy, font_scale)
    write_video(out, dst, fps=fps)


def shuffle_frames(src, dst, seed=42):
    frames, fps = read_video(src)
    order = list(range(len(frames)))
    random.Random(seed).shuffle(order)
    write_video(frames[order], dst, fps=fps)


def reverse_frames(src, dst, **_):
    frames, fps = read_video(src)
    write_video(frames[::-1], dst, fps=fps)


OPS = {
    "gau": gaussian_noise,
    "mb":  motion_blur,
    "cmp": compress,
    "cap": caption_injection,
    "sub": subtitle_corruption,
    "shu": shuffle_frames,
    "rev": reverse_frames,
}
