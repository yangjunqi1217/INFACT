import numpy as np


def sample_indices(total, n):
    if n >= total:
        return list(range(total))
    step = total / n
    return [min(total - 1, int(step * i + step / 2)) for i in range(n)]


def read_video(path, n_frames=None):
    from decord import VideoReader, cpu
    vr = VideoReader(str(path), ctx=cpu(0))
    idx = list(range(len(vr))) if n_frames is None else sample_indices(len(vr), n_frames)
    frames = vr.get_batch(idx).asnumpy().astype(np.uint8)
    fps = float(vr.get_avg_fps()) or 24.0
    return frames, fps


def write_video(frames, path, fps=24.0, crf=18):
    import imageio.v2 as iio
    w = iio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
    )
    try:
        for f in frames:
            w.append_data(f)
    finally:
        w.close()
