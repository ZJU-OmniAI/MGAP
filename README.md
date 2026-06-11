# MGAP: Manifold-Guided Adaptive Projection

[![Venue: ICML 2026](https://img.shields.io/badge/Venue-ICML%202026-blue.svg)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)]([#](https://arxiv.org/abs/2606.09859)) Official implementation for the ICML 2026 paper:
**"Mitigating Manifold Departure: Uncertainty-Aware Subspace Rectification for Trustworthy MLLM Decoding"**

MGAP is a geometry-aware, training-free decoding method that mitigates object hallucinations in Multimodal Large Language Models (MLLMs). By constructing a label-free language prior subspace and applying consistency-aware adaptive projection, MGAP suppresses hallucinations without disrupting the model's semantic manifold.


**This work is built upon the code framework of [HalTrapper](https://github.com/SooLab/HalTrapper)** (ICCV 2025). We sincerely thank the HalTrapper authors for open-sourcing their implementation and evaluation pipeline.




## Setup

### Environment setup

Our code requires **Python ≥ 3.9** (3.10 recommended). For **LLaVA 1.5**, we use **`transformers==4.37.2`**. Due to API changes across `transformers` versions, other versions may cause errors. Version checks are included in some modules.

| Model      | `transformers` version |
| ---------- | ---------------------- |
| LLaVA 1.5  | **4.37.2**             |
| Qwen VL    | 4.32.0                 |
| MiniGPT-4  | 4.30.0                 |
| Qwen2 VL   | 4.45.0                 |
| Janus Pro  | 4.48.3                 |
| Qwen3 VL   | 4.57.3                 |

Additionally, to evaluate CHAIR and AMBER, install the following:

```bash
pip install spacy nltk "numpy<2"
python -m spacy download en_core_web_lg
```


### Path setup

Paths are **relative to the MGAP root** by default. You can either place files in the directories below or override them via CLI flags / `playground/path_table.py`.

| Resource        | Default path                    | CLI override        |
| --------------- | ------------------------------- | ------------------- |
| LLaVA weights   | `models/llava-v1.5-7b/`         | `--model-path`      |
| COCO val2014    | `data/coco/val2014/`            | `--coco-img-dir`    |
| POPE annotations| `data/pope/`                    | `--pope-dir`        |
| MathVista       | `data/mathvista/test.parquet`   | `--mathvista-file`  |
| AMBER           | `data/amber/`                   | `path_table.py`     |
| Outputs         | `results/`                      | `--output-dir`      |

See `models/README.md` and `data/README.md` for details.

#### COCO & AMBER

To evaluate **CHAIR** and **AMBER**, download:

- [COCO Dataset](https://cocodataset.org/) — use the `val2014/` folder with images directly inside it.
- [AMBER Repository](https://github.com/junyangwang0410/AMBER) — use the `data/` folder from that repo; images are expected under `data/image/`.

#### POPE & MathVista

Place POPE JSON files (e.g. `coco_pope_random.json`) under `data/pope/`. For MathVista, place `test.parquet` under `data/mathvista/`, or provide a sibling `.json` / `.jsonl` fallback if parquet engines are unavailable.

## Evaluation

### Start of evaluation

**CHAIR** (greedy decoding, fixed 500 images):

```bash
python decontext.py \
    --model llava \
    --method [method] \
    --eval chair \
    --fixed True
```

`--fixed True` uses a fixed set of 500 questions instead of random sampling.

**AMBER** (generative subset):

```bash
python decontext.py \
    --model llava \
    --method [method] \
    --eval amber \
    --split g \
    --change-prompt True
```

`--split g` evaluates the generative subset only.

**POPE** (three splits: random / popular / adversarial):

```bash
python decontext.py \
    --model llava \
    --method [method] \
    --eval pope
```


### Methods

| `[method]`   | Description                                      |
| ------------ | ------------------------------------------------ |
| `baseline`   | Vanilla model                                    |
| `vcd`        | [Visual Contrastive Decoding](https://github.com/DAMO-NLP-SG/VCD) |
| `icd`        | [Instruction Contrastive Decoding](https://github.com/hillzhang1999/ICD) |
| `pai`        | [Paying More Attention to Image](https://github.com/LALBJ/PAI) |
| `code`       | [CODE](https://github.com/IVY-LVLM/CODE)         |
| `haltrapper` | [HalTrapper](https://github.com/SooLab/HalTrapper)              |
| `mgap`       | MGAP method (`ours` /  `urs_vcd` aliases) |

Add `--sample` for nucleus sampling, or `--num_beams 5` for beam search.

### Output and configuration logging

Results are written under `results/` (or `--output-dir`) as `.jsonl` files. A matching `-config.yaml` is saved alongside each run.

To **re-score existing `.jsonl` outputs**:

**CHAIR**

```bash
python -m playground.eval \
    [path/to/model-outputs.jsonl] \
    --eval chair \
    --fixed True
```

**AMBER**

```bash
python -m playground.eval \
    [path/to/model-outputs.jsonl] \
    --eval amber \
    --split g \
    --change-prompt True
```

### Candidate cache

Methods such as `haltrapper` and `code` build reusable hallucination candidates per image. Candidates are stored under the `cache/` folder and speed up later runs on the same images. Delete `cache/` to clear stored data.

## Acknowledgements

We thank the authors of **[HalTrapper](https://github.com/SooLab/HalTrapper)** for their open-source framework, which this repository extends.

Our implementation also incorporates or modifies code from (in no particular order):
- [junyangwang0410/AMBER](https://github.com/junyangwang0410/AMBER)
- [IVY-LVLM/CODE](https://github.com/IVY-LVLM/CODE)
- [hillzhang1999/ICD](https://github.com/hillzhang1999/ICD)
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- [LALBJ/PAI](https://github.com/LALBJ/PAI)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [DAMO-NLP-SG/VCD](https://github.com/DAMO-NLP-SG/VCD)

## Citation

If you find our work or this code useful for your research, please cite our paper:

```bibtex
@inproceedings{zhuang2026mitigating,
    title     = {Mitigating Manifold Departure: Uncertainty-Aware Subspace Rectification for Trustworthy MLLM Decoding},
    author    = {Zhuang, Yingxuan and Yang, Jingxiao and Pan, Miao and Tan, Cheng and Cai, Yuxiang and Tan, Siwei and Zhi, Chen and Zhang, Xuhong and Yin, Jianwei and Chen, Jintao},
    booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
    year      = {2026}
    url       = {https://openreview.net/forum?id=LInDSHWGMK},
}
