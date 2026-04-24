# INFACT

Code and data pipeline for *INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs* (ACL 2026 main).

Junqi Yang, Yuecong Min, Jie Zhang\*, Shiguang Shan, Xilin Chen. ICT, CAS.

## What's here

- 9,800 multiple-choice items across two tracks: faithfulness and factuality.
- 12 + 12 fine-grained categories (see `docs/TAXONOMY.md`).
- Four evaluation modes: **Base**, **Visual Degradation** (`gau` / `mb` / `cmp`), **Evidence Corruption** (`cap` / `sub` / `adv`), **Temporal Intervention** (`shu` / `rev`).
- Metrics: Resist Rate (RR) for label-preserving perturbations, Temporal Sensitivity Score (TSS) for order-destroying ones, each paired with the model's Base prediction.
- `adv` runs MI-FGSM separately against three proxies {InternVL3-8B, Qwen3-VL-8B, VideoMAE-large}; `RR_adv` is the mean of the 3 per-proxy RRs.

## Status

Annotations (`data/questions/*.jsonl`) and synthetic Know-WHY videos are **not yet released**; download link will appear here once the camera-ready artifact is finalised. Real-video splits point to upstream datasets — you must obtain those separately (see *Data* below).

The code has been sanity-checked but not all 14 model wrappers are end-to-end tested against the final release of each backbone repo. File an issue if a loader breaks on your setup.

## Install

```
pip install -r requirements.txt
```

`torch` / `transformers` are needed for local model inference; install a build matching your CUDA. API models (`gpt5_1`, `gemini3_flash`) need `openai` / `google-genai`.

Five backbones are not on PyPI and need to be installed from their own repos — each pins its own `transformers` version, so evaluate them in separate envs:

- VideoLLaMA2 — https://github.com/DAMO-NLP-SG/VideoLLaMA2 (`transformers~=4.40`)
- PLLaVA — https://github.com/magic-research/PLLaVA (imports via `models.pllava`, not `pllava`)
- NVILA — https://github.com/NVlabs/VILA (package name `llava`; isolate from any other `llava`)
- ShareGPT4Video — https://github.com/ShareGPT4Omni/ShareGPT4Video (`transformers==4.37.2`)
- Tarsier — https://github.com/bytedance/tarsier (branch `tarsier`, `transformers==4.44.2`)

Qwen3-VL needs `transformers>=4.57`.

## Data

Annotations follow:

```json
{"id": "...", "track": "faithfulness", "category": "3.4_temporal_localization",
 "level": 3, "source_dataset": "TOMATO", "video": "videos/tomato/xxx.mp4",
 "question": "...", "options": [...], "answer_idx": 2, "order_sensitive": true}
```

Upstream sources (obtain + agree to each one's license separately):

| Source | Used for | Link |
|---|---|---|
| MVBench | faithfulness | https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2 |
| Video-MME | faithfulness | https://github.com/MME-Benchmarks/Video-MME |
| TOMATO | faithfulness | https://github.com/yale-nlp/TOMATO |
| CityGuessr68k | factuality / domain | https://github.com/ParthPK/CityGuessr |
| ViMUL-Bench | factuality / domain | https://github.com/mbzuai-oryx/ViMUL |
| COIN | factuality / procedural | https://github.com/coin-dataset/annotations |
| MedVidQA | factuality / procedural (clinical) | https://github.com/deepaknlp/MedVidQACL |

`cap` reads an irrelevant-caption pool from `data/meta/irrelevant_captions.txt` (one line per caption; a small sample is at `data/meta/irrelevant_captions.sample.txt`). Per-item `meta.misleading_caption` and `meta.transcript` must be produced during benchmark construction (not scripted in this repo).

## Run

```
# 1. perturb
python scripts/perturb.py --split faithfulness --ops gau mb cmp cap sub shu rev
python scripts/perturb.py --split faithfulness --ops adv        # GPU

# 2. infer (per model)
python scripts/eval.py --model qwen3_vl_8b --split faithfulness \
    --out runs/qwen3_vl_8b/faithfulness

# 3. aggregate
python scripts/score.py --runs runs/qwen3_vl_8b/faithfulness --split faithfulness
```

`python scripts/eval.py --list-models` prints the 14 registered names.

## Licensing

- **Code** in this repo: MIT (see `LICENSE`).
- **Annotations** (once released): CC-BY-4.0; cite this paper and all applicable upstream sources.
- **Real videos**: not redistributed by us; retain the license of their original source dataset.
- **Synthetic Know-WHY clips** (once released): subject to the generator's terms (OpenAI Sora, Alibaba Wan2.5, Google Veo 3). We release prompts + seeds so users can regenerate under their own accounts when redistribution is restricted.
- **Proprietary-model outputs** (GPT-5.1, Gemini 3 Flash): the numerical results reported in the paper were produced for academic benchmarking and are not intended for training competing models.

## Ethics & intended use

- **MedVidQA** contains medical / clinical instructional content. INFACT's Know-HOW clinical subset is intended for benchmarking language-grounded procedural reasoning **only**. These videos and QA pairs are **not** clinical guidance and must not be used to train, tune, or deploy any medical decision-support system.
- **CityGuessr68k** ties images to geographic locations. Use within the bounds of its original license; do not attempt to de-anonymise uploaders.
- **Synthetic Know-WHY** probes intentionally violate physical laws to test factuality. Do not mine them as training data for generative models.

## Citation

If you use INFACT, please cite both this paper and all upstream sources you use videos / QA from.

```
@inproceedings{yang2026infact,
  title     = {INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs},
  author    = {Yang, Junqi and Min, Yuecong and Zhang, Jie and Shan, Shiguang and Chen, Xilin},
  booktitle = {Proceedings of ACL},
  year      = {2026}
}
```

Contact: `yangjunqi23@mails.ucas.ac.cn` (first author), `zhangjie@ict.ac.cn` (corresponding).
