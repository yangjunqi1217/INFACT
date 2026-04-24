import base64
import io
import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from .prompts import build_prompt, parse_answer
from .video import read_video


MODELS = {
    "videollama2_7b":    "DAMO-NLP-SG/VideoLLaMA2-7B",
    "qwen2_5_vl_7b":     "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_32b":    "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen3_vl_8b":       "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_32b":      "Qwen/Qwen3-VL-32B-Instruct",
    "internvl3_8b":      "OpenGVLab/InternVL3-8B",
    "internvl3_5_8b":    "OpenGVLab/InternVL3_5-8B",
    "sharegpt4video_8b": "Lin-Chen/sharegpt4video-8b",
    "nvila_8b":          "Efficient-Large-Model/NVILA-8B",
    "pllava_13b":        "ermu2001/pllava-13b",
    "pllava_34b":        "ermu2001/pllava-34b",
    "tarsier_34b":       "omni-research/Tarsier-34b",
    "gpt5_1":            "gpt-5.1-2025-11-13",
    "gemini3_flash":     "gemini-3-flash-preview",
}


def answer(name, video, question, options, n_frames=16, device="cuda"):
    prompt = build_prompt(question, options)
    if name.startswith("qwen"):
        text = _qwen(name, video, prompt, n_frames, device)
    elif name.startswith("internvl"):
        text = _internvl(name, video, prompt, n_frames, device)
    elif name in ("pllava_13b", "pllava_34b"):
        text = _pllava(name, video, prompt, n_frames, device)
    elif name == "sharegpt4video_8b":
        text = _sharegpt4v(video, prompt, n_frames, device)
    elif name == "tarsier_34b":
        text = _tarsier(video, prompt, n_frames, device)
    elif name == "videollama2_7b":
        text = _videollama2(video, prompt, n_frames)
    elif name == "nvila_8b":
        text = _nvila(video, prompt, n_frames, device)
    elif name == "gpt5_1":
        text = _openai(video, prompt, n_frames)
    elif name == "gemini3_flash":
        text = _gemini(video, prompt)
    else:
        raise ValueError(f"unknown model: {name}")
    return {"response": text, "pred_idx": parse_answer(text, len(options))}


@lru_cache
def _qwen_bundle(name, device):
    from transformers import AutoProcessor
    if name.startswith("qwen2_5"):
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    else:
        from transformers import Qwen3VLForConditionalGeneration as cls
    ckpt = MODELS[name]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = cls.from_pretrained(ckpt, torch_dtype=dtype, device_map=device).eval()
    return model, AutoProcessor.from_pretrained(ckpt)


def _qwen(name, video, prompt, n_frames, device):
    from qwen_vl_utils import process_vision_info
    model, proc = _qwen_bundle(name, device)
    msg = [{"role": "user", "content": [
        {"type": "video", "video": str(video), "nframes": n_frames},
        {"type": "text",  "text": prompt},
    ]}]
    chat = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msg)
    inp = proc(text=[chat], images=imgs, videos=vids, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=16, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[-1]:], skip_special_tokens=True)[0]


@lru_cache
def _intern_bundle(name, device):
    from transformers import AutoModel, AutoTokenizer
    ckpt = MODELS[name]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    return model, AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)


def _internvl(name, video, prompt, n_frames, device):
    import torchvision.transforms as T

    model, tok = _intern_bundle(name, device)
    frames, _ = read_video(video, n_frames=n_frames)
    tfm = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil = [Image.fromarray(f) for f in frames]
    pixel = torch.stack([tfm(f) for f in pil]).to(device, dtype=next(model.parameters()).dtype)
    q = "".join(f"Frame{i+1}: <image>\n" for i in range(len(pil))) + prompt
    resp = model.chat(
        tokenizer=tok, pixel_values=pixel, question=q,
        generation_config={"max_new_tokens": 16, "do_sample": False},
        num_patches_list=[1] * len(pil),
    )
    return resp if isinstance(resp, str) else resp[0]


@lru_cache
def _videollama2_bundle():
    from videollama2 import model_init
    from videollama2.utils import disable_torch_init
    disable_torch_init()
    return model_init(MODELS["videollama2_7b"])


def _videollama2(video, prompt, n_frames):
    from videollama2 import mm_infer
    model, proc, tok = _videollama2_bundle()
    video_tensor = proc["video"](str(video), num_frames=n_frames)
    return mm_infer(video_tensor, prompt, model=model, tokenizer=tok,
                    do_sample=False, modal="video")


@lru_cache
def _pllava_bundle(name, device):
    from transformers import AutoProcessor
    from models.pllava import PllavaForConditionalGeneration
    ckpt = MODELS[name]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    proc = AutoProcessor.from_pretrained(ckpt)
    model = PllavaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=dtype).to(device).eval()
    return proc, model


def _pllava(name, video, prompt, n_frames, device):
    proc, model = _pllava_bundle(name, device)
    frames, _ = read_video(video, n_frames=n_frames)
    inp = proc(text=prompt, images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=16, do_sample=False)
    return proc.batch_decode(out, skip_special_tokens=True)[0]


@lru_cache
def _nvila_bundle():
    import llava
    return llava.load(MODELS["nvila_8b"])


def _nvila(video, prompt, n_frames, device):
    from llava.media import Video
    model = _nvila_bundle()
    model.config.num_video_frames = n_frames
    model.config.fps = 0.0
    return model.generate_content(
        [Video(str(video)), prompt],
        generation_config=model.default_generation_config,
    )


_SGPT4V_PRE = (
    "The provided image arranges keyframes from a video in a grid view, "
    "keyframes are separated with white bands. Answer concisely with overall "
    "content and context of the video, highlighting any significant events, "
    "characters, or objects that appear throughout the frames."
)


def _frame_grid(frames, sep=50):
    T, H, W, _ = frames.shape
    side = int(np.ceil(np.sqrt(T)))
    bg = np.full((side * H + (side + 1) * sep, side * W + (side + 1) * sep, 3), 255, dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, side)
        y, x = sep + r * (H + sep), sep + c * (W + sep)
        bg[y:y + H, x:x + W] = f
    return bg


@lru_cache
def _sharegpt4v_bundle(device):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    disable_torch_init()
    ckpt = MODELS["sharegpt4video_8b"]
    tok, model, proc, _ = load_pretrained_model(
        ckpt, None, get_model_name_from_path(ckpt), device_map="cpu",
    )
    return tok, model.to(device).eval(), proc


def _sharegpt4v(video, prompt, n_frames, device):
    from decord import VideoReader, cpu
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    tok, model, proc = _sharegpt4v_bundle(device)
    vr = VideoReader(str(video), ctx=cpu(0), num_threads=1)
    T = len(vr)
    seg = (T - 1) / n_frames
    idx = np.array([int(seg / 2 + round(seg * i)) for i in range(n_frames)])
    frames = vr.get_batch(idx).asnumpy()
    grid = Image.fromarray(_frame_grid(frames)).convert("RGB")

    conv = conv_templates["llava_llama_3"].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{_SGPT4V_PRE}{prompt}")
    conv.append_message(conv.roles[1], None)
    text = conv.get_prompt()

    image_tensor = process_images([grid], proc, model.config)[0]
    input_ids = tokenizer_image_token(text, tok, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt").unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids, images=image_tensor.to(device=device, dtype=torch.float16),
            image_sizes=[grid.size], do_sample=False, max_new_tokens=16,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


@lru_cache
def _tarsier_bundle():
    from models.modeling_tarsier import TarsierForConditionalGeneration, LlavaConfig
    from dataset.processor import Processor
    ckpt = MODELS["tarsier_34b"]
    processor = Processor(ckpt, max_n_frames=16)
    config = LlavaConfig.from_pretrained(ckpt, trust_remote_code=True)
    model = TarsierForConditionalGeneration.from_pretrained(
        ckpt, config=config, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True,
    ).eval()
    return processor, model


def _tarsier(video, prompt, n_frames, device):
    processor, model = _tarsier_bundle()
    processor.max_n_frames = n_frames
    inputs = processor(f"<video>\n{prompt}", str(video), edit_prompt=True, return_prompt=True)
    inputs.pop("prompt", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
    out = model.generate(**inputs, do_sample=False, max_new_tokens=16,
                         top_p=1.0, temperature=0, use_cache=True)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"][0].shape[0]:], skip_special_tokens=True,
    )


@lru_cache
def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _jpg_b64(frame):
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _openai(video, prompt, n_frames):
    frames, _ = read_video(video, n_frames=n_frames)
    content = [{"type": "text", "text": prompt}]
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_jpg_b64(f)}", "detail": "low"},
        })
    resp = _openai_client().chat.completions.create(
        model=MODELS["gpt5_1"],
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=16, temperature=0.0,
    )
    return resp.choices[0].message.content or ""


@lru_cache
def _gemini_client():
    from google import genai
    return genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options={"api_version": "v1alpha"},
    )


def _gemini(video, prompt):
    c = _gemini_client()
    up = c.files.upload(file=str(video))
    r = c.models.generate_content(
        model=MODELS["gemini3_flash"],
        contents=[up, prompt],
        config={"temperature": 0.0, "max_output_tokens": 16,
                "media_resolution": "media_resolution_medium"},
    )
    return (r.text or "").strip()
