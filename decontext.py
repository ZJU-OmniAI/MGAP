import os
import sys
import argparse
import torch
import numpy as np
import json
import io
import re
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Optional

from mgap_paths import MGAP_ROOT, mgap_path

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

from models_modified import LlavaModified
from parsers import AmberParser, ChairParser
from methods_utils.cache_table import ContextCDCandidates
from playground._utils._path import save_structured_file, load_structured_file
from playground import get_eval_benchmark_from_args
from playground._utils._colors import *
from playground._utils._seed import seed_everything
from playground.chair.chair import CHAIR

MATHVISTA_COT_SUFFIX_RE = re.compile(
    r"\s*You FIRST think about the reasoning process as an internal monologue and then provide the final answer\.?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)

def normalize_structured_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def extract_structured_text(value) -> str:
    value = normalize_structured_value(value)

    if isinstance(value, dict):
        for key in ("content", "text", "value"):
            if value.get(key) is not None:
                return str(value[key])
        return ""

    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            item_text = extract_structured_text(item)
            if item_text:
                parts.append(item_text)
        return "\n".join(parts)

    if value is None:
        return ""

    return str(value)

def sanitize_mathvista_prompt(prompt_value) -> str:
    prompt_text = extract_structured_text(prompt_value).strip()
    prompt_text = re.sub(
        r"^\s*(?:<image>|<image-placeholder>|<Image>)+\s*",
        "",
        prompt_text,
        count=1,
        flags=re.IGNORECASE,
    )
    prompt_text = MATHVISTA_COT_SUFFIX_RE.sub("", prompt_text).strip()
    return prompt_text

def extract_mathvista_ground_truth(item: dict) -> str:
    reward_model = normalize_structured_value(item.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"]).strip()

    extra_info = normalize_structured_value(item.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
        return str(extra_info["answer"]).strip()

    return ""

def extract_mathvista_prompt(item: dict) -> str:
    prompt_text = sanitize_mathvista_prompt(item.get("prompt"))
    if prompt_text:
        return prompt_text

    extra_info = normalize_structured_value(item.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("question"):
        question = str(extra_info["question"]).strip()
        gt_answer = extract_mathvista_ground_truth(item)
        if re.fullmatch(r"[A-Z]", gt_answer):
            return (
                "Please answer the question and provide the correct option letter "
                f"at the end.\nQuestion: {question}"
            )
        return f"Please answer the question and provide the final value at the end.\nQuestion: {question}"

    return "Describe the image."

def materialize_mathvista_image(images_value, sample_id: int) -> tuple[str, bool]:
    images_value = normalize_structured_value(images_value)
    image_item = images_value[0] if isinstance(images_value, (list, tuple)) else images_value
    image_item = normalize_structured_value(image_item)

    image_bytes = None
    if isinstance(image_item, dict):
        if image_item.get("path"):
            return str(image_item["path"]), False
        if image_item.get("file_name"):
            return str(image_item["file_name"]), False
        image_bytes = image_item.get("bytes")
    elif isinstance(image_item, (bytes, bytearray)):
        image_bytes = image_item
    elif isinstance(image_item, str):
        return image_item, False

    if image_bytes is None:
        raise ValueError(f"Unsupported MathVista image payload for sample {sample_id}.")

    temp_dir = Path("/tmp/mathvista_images")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"mathvista_{os.getpid()}_{sample_id}.jpg"
    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(temp_path, format="JPEG")
    return str(temp_path), True

def extract_mathvista_prediction(response: str, ground_truth: str) -> str:
    response = (response or "").strip()
    ground_truth = (ground_truth or "").strip()

    if re.fullmatch(r"[A-Z]", ground_truth):
        option_matches = re.findall(r"\b([A-Z])\b", response.upper())
        if option_matches:
            return option_matches[-1]
        return response.upper().strip()

    number_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", response)
    if number_matches:
        return number_matches[-1].replace(",", "")

    return response.strip()

def mathvista_answer_is_correct(pred_answer: str, ground_truth: str) -> bool:
    pred_answer = (pred_answer or "").strip()
    ground_truth = (ground_truth or "").strip()

    if not ground_truth:
        return False

    if re.fullmatch(r"[A-Z]", ground_truth):
        return pred_answer.upper() == ground_truth.upper()

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", ground_truth) and re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?", pred_answer
    ):
        return abs(float(pred_answer) - float(ground_truth)) < 1e-6

    return pred_answer.lower() == ground_truth.lower()

def extract_calibration_text(item) -> Optional[str]:
    text = None
    if isinstance(item, dict):
        text = item.get("text") or item.get("question") or item.get("instruction")

        if text is None:
            extra_info = normalize_structured_value(item.get("extra_info"))
            if isinstance(extra_info, dict):
                text = extra_info.get("question")

        if text is None and item.get("prompt") is not None:
            text = extract_mathvista_prompt(item)

    if text is None:
        return None

    text = str(text).strip()
    return text or None

def load_mathvista_records(file_path: Path) -> list[dict]:
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".parquet":
        try:
            import pandas as pd

            mathvista_df = pd.read_parquet(file_path)
            return mathvista_df.to_dict(orient="records")
        except ImportError as e:
            fallback_candidates = [file_path.with_suffix(".json"), file_path.with_suffix(".jsonl")]
            for candidate in fallback_candidates:
                if candidate.exists():
                    print_warning(
                        f"Parquet engine unavailable in current environment; using MathVista fallback file: {candidate}"
                    )
                    records = load_structured_file(candidate)
                    if not isinstance(records, list):
                        raise ValueError(f"MathVista fallback file must contain a list of records: {candidate}")
                    return records

            raise ImportError(
                "Unable to read MathVista parquet because no parquet engine is installed. "
                "Install pyarrow/fastparquet, or place a sibling test.json/test.jsonl fallback next to the parquet file."
            ) from e

    if ext in {".json", ".jsonl"}:
        records = load_structured_file(file_path)
        if not isinstance(records, list):
            raise ValueError(f"MathVista file must contain a list of records: {file_path}")
        return records

    raise ValueError(f"Unsupported MathVista file format: {file_path}")

class BiasSubspaceManager:
    def __init__(self, device):
        self.device = device
        self.bias_components = None

    def get_hidden_state_only(self, model, tokenizer, image_processor, text, conv_mode):

        dummy_image = Image.new('RGB', (336, 336), (0, 0, 0))
        qs = text
        if DEFAULT_IMAGE_TOKEN not in qs:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        image_tensor = image_processor.preprocess(dummy_image, return_tensors='pt')['pixel_values'][0]

        input_ids = input_ids.unsqueeze(0).to(model.device)
        image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, images=image_tensor, output_hidden_states=True, return_dict=True)

        return outputs.hidden_states[-1][:, -1, :]

    def calibrate(self, model_wrapper, benchmark_data, args, n_samples=50):
        """
        Calibrate using benchmark data.
        benchmark_data: list (POPE) or other iterable
        """
        print_note(f"🔄 [Calibration] Building Hallucination Subspace (Samples: {n_samples})...")

        hf_model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
        image_processor = model_wrapper.image_processor

        calibration_set = []
        count = 0

        iterator = benchmark_data
        if hasattr(benchmark_data, 'data'):
            iterator = benchmark_data.data

        for item in iterator:
            if count >= n_samples: break

            text = extract_calibration_text(item)

            if text is None:
                text = "Describe the image."

            if text:
                calibration_set.append(text)
                count += 1

        blind_states = []
        conv_mode = getattr(args, "conv_mode", "vicuna_v1")

        for text in tqdm(calibration_set, desc="Collecting Bias Vectors"):
            h_blind = self.get_hidden_state_only(
                hf_model, tokenizer, image_processor,
                text, conv_mode
            )
            blind_states.append(h_blind.cpu().numpy())

        if len(blind_states) == 0:
            print_warning("No calibration data collected! Subspace will be None.")
            return

        X = np.vstack(blind_states)
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        U, S, Vh = np.linalg.svd(X_centered.astype(np.float32), full_matrices=False)

        K = args.subspace_dim
        self.bias_components = torch.tensor(Vh[:K, :]).to(self.device).float()
        print_note(f"✅ Subspace constructed (Dim={K})")

    def project_vector(self, h):
        if self.bias_components is None: return torch.zeros_like(h)
        h = h.float()
        coeffs = torch.matmul(h, self.bias_components.T)
        bias_proj = torch.matmul(coeffs, self.bias_components)
        return bias_proj.half()

def run_pope_eval(result_file, annotation_file):
    print_note(f"📊 Evaluating POPE: {result_file}")

    with open(annotation_file, 'r') as f:
        ref_labels = [json.loads(line) for line in f]
    with open(result_file, 'r') as f:
        res_labels = [json.loads(line) for line in f]

    results = {'TP': 0, 'TN': 0, 'FP':0, 'FN': 0}

    res_dict = {item['question_id']: item for item in res_labels}

    num_sample = 0
    for ref in ref_labels:
        idx = ref["question_id"]
        if idx not in res_dict:
            continue

        res = res_dict[idx]
        num_sample += 1

        ref_label = ref["label"].lower().strip()
        res_text = res["text"].lower().strip()

        if ref_label == 'yes':
            if 'yes' in res_text:
                results['TP'] += 1
            else:
                results['FN'] += 1
        else:
            if 'no' in res_text or 'not' in res_text:
                results['TN'] += 1
            else:
                results['FP'] += 1

    if num_sample == 0:
        return 0, 0, 0, 0

    Accuracy = (results['TP'] + results['TN']) / num_sample

    if (results['TP'] + results['FP']) > 0:
        Precision = results['TP'] / (results['TP'] + results['FP'])
    else:
        Precision = 0

    if (results['TP'] + results['FN']) > 0:
        Recall = results['TP'] / (results['TP'] + results['FN'])
    else:
        Recall = 0

    if (Precision + Recall) > 0:
        F1 = 2 * Precision * Recall / (Precision + Recall)
    else:
        F1 = 0

    print(f"   Accuracy : {Accuracy:.4f}")
    print(f"   Precision: {Precision:.4f}")
    print(f"   Recall   : {Recall:.4f}")
    print(f"   F1 Score : {F1:.4f}")
    print("-" * 30)

    return Accuracy, Precision, Recall, F1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava", choices=["llava"])
    parser.add_argument("--model-path", type=str, default=None, help="LLaVA weights directory (default: models/llava-v1.5-7b)")
    parser.add_argument("--method", type=str, default="haltrapper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(mgap_path("results")),
        help="Directory for inference outputs",
    )

    parser.add_argument("--eval", type=str, default="chair", choices=["chair", "pope", "amber", "mathvista"])

    parser.add_argument(
        "--pope-dir",
        type=str,
        default=str(mgap_path("data", "pope")),
        help="Directory containing coco_pope_xxx.json files",
    )
    parser.add_argument(
        "--coco-img-dir",
        type=str,
        default=str(mgap_path("data", "coco", "val2014")),
        help="COCO val2014 image dir",
    )
    parser.add_argument(
        "--mathvista-file",
        type=str,
        default=str(mgap_path("data", "mathvista", "test.parquet")),
        help="MathVista parquet file",
    )
    parser.add_argument("--mathvista-calibration-source", type=str, default="pope", choices=["pope", "mathvista"], help="Calibration source for MathVista MGAP subspace")
    parser.add_argument("--mathvista-calibration-file", type=str, default=None, help="Optional calibration file for MathVista MGAP. Defaults to POPE random split under --pope-dir")

    parser.add_argument("--log-file-path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)

    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat_mode", type=str, default="continuous")
    parser.add_argument("--ee_threshold", type=float, default=None)
    parser.add_argument("--ig_strategy", type=str, default="cos_sim")
    parser.add_argument("--ig_threshold", type=float, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--sep", type=str, default=None)

    parser.add_argument("--noise_step", type=int, default=500)

    parser.add_argument("--pai_alpha", type=float, default=0.5)

    parser.add_argument("--subspace_dim", type=int, default=3, help="Dimension of bias subspace")
    parser.add_argument("--urs_alpha", type=float, default=0.75, help="Alpha for MGAP")
    parser.add_argument("--gate_sensitivity", type=float, default=0.6, help="Entropy gate sensitivity")
    parser.add_argument("--cos_threshold", type=float, default=0.25, help="Cosine protection threshold")
    parser.add_argument("--norm_restoration", action="store_true", default=True, help="Enable Norm Restoration")
    parser.add_argument("--calibration_samples", type=int, default=50, help="Number of samples for calibration")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")

    parser.add_argument("--no_protection", action="store_true", help="Disable protection mechanism (ablation)")
    parser.add_argument("--no_gate", action="store_true", help="Disable gate mechanism (ablation)")

    args, remain_args = parser.parse_known_args()

    seed_everything(args.seed)

    method: str = args.method.lower()
    if method in ("ours", "clearsight", "urs_vcd"):
        method = "mgap"

    model_name: str = args.model.lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    if model_path is None:
        model_path = str(mgap_path("models", "llava-v1.5-7b"))

    print_note(f"Using model {model_name}.")
    print_note(f"Using method {method}.")
    print_note(f"Model path: {model_path}")
    print_note(f"Output dir: {output_dir}")

    if model_name == "llava":
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"LLaVA weights not found: {model_path}\n"
                f"Place liuhaotian/llava-v1.5-7b at {mgap_path('models', 'llava-v1.5-7b')}, "
                "or set --model-path."
            )
        model = LlavaModified(model_path=model_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    kwargs = {
        "temperature": args.temperature if args.sample else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
    }

    method_kwargs = {}

    if method == "mgap":

        method_kwargs = {
            "subspace_dim": args.subspace_dim,
            "alpha": args.urs_alpha,
            "gate_sensitivity": args.gate_sensitivity,
            "cos_threshold": args.cos_threshold,
            "norm_restoration": args.norm_restoration,
            "protection_ablation": not args.no_protection,
            "gate_ablation": not args.no_gate
        }
    elif method == "pai":
        method_kwargs = {"pai_alpha": args.pai_alpha}

    if args.pai_alpha is not None:
        method_kwargs["pai_alpha"] = args.pai_alpha

    if args.eval == "pope":
        print_note(f"🚀 Starting POPE Evaluation (Method: {method})")

        pope_files = [
            "coco_pope_random.json",
            "coco_pope_popular.json",
            "coco_pope_adversarial.json"
        ]

        if not os.path.exists(args.pope_dir):

            if os.path.exists("./POPE/coco"):
                args.pope_dir = "./POPE/coco"
            else:
                raise FileNotFoundError(f"POPE directory not found at {args.pope_dir}")

        summary = {}

        if method == "mgap":
            calib_file = os.path.join(args.pope_dir, "coco_pope_random.json")

            calib_data = [json.loads(line) for line in open(calib_file, 'r')]

            subspace_manager = BiasSubspaceManager(model.model.device)
            subspace_manager.calibrate(model, calib_data, args, n_samples=args.calibration_samples)
            model.subspace_manager = subspace_manager

        for pope_file in pope_files:
            subset_name = pope_file.replace("coco_pope_", "").replace(".json", "")
            print_note(f"\n>>> Running POPE Subset: {subset_name}")

            file_path = os.path.join(args.pope_dir, pope_file)
            if not os.path.exists(file_path):
                print_warning(f"File not found: {file_path}, skipping.")
                continue

            questions = [json.loads(line) for line in open(file_path, 'r')]

            ablation_suffix = ""
            if method == "mgap":
                if args.no_protection:
                    ablation_suffix += "_noprot"
                if args.no_gate:
                    ablation_suffix += "_nogate"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_file = str(
                output_dir / f"{model.name}-pope-{subset_name}-{method}{ablation_suffix}_{timestamp}.jsonl"
            )
            out_record_file = str(
                output_dir / f"{model.name}-pope-{subset_name}-{method}{ablation_suffix}_{timestamp}_records.npz"
            )

            all_records = []

            with open(out_file, 'w') as ans_file:
                for i, item in enumerate(tqdm(questions)):
                    image_file = item["image"]
                    question = item["text"]
                    question_id = item["question_id"]
                    label = item["label"]

                    image_path = os.path.join(args.coco_img_dir, image_file)

                    current_kwargs = kwargs.copy()
                    current_method_kwargs = method_kwargs.copy()
                    if method == "mgap":
                        current_method_kwargs['record_internals'] = True

                    response, _, _, first_token_record = model.submit(
                        prompt=question,
                        image=image_path,
                        method=method,
                        question_id=question_id,

                        **current_kwargs,
                        **current_method_kwargs
                    )

                    if first_token_record is not None:
                        record = {
                            'question_id': question_id,
                            'label': label,
                            'response': response,
                            'image_file': image_file,
                            **first_token_record
                        }
                        all_records.append(record)

                    ans_file.write(json.dumps({
                        "question_id": question_id,
                        "prompt": question,
                        "text": response,
                        "model_id": model_name,
                        "image": image_file,
                        "metadata": {}
                    }) + "\n")
                    ans_file.flush()

            acc, prec, rec, f1 = run_pope_eval(out_file, file_path)
            summary[subset_name] = {"Acc": acc, "F1": f1, "Prec": prec, "Rec": rec}

        print("\n" + "="*40)
        print("🏆 POPE Final Summary")
        print("="*40)
        for name, metrics in summary.items():
            print(f"{name.upper():<12}: Acc={metrics['Acc']:.2f}, F1={metrics['F1']:.2f}, Prec={metrics['Prec']:.2f}, Rec={metrics['Rec']:.2f}")
        print("="*40)

    elif args.eval == "mathvista":
        print_note(f"🚀 Starting MathVista Evaluation (Method: {method})")

        mathvista_file = Path(args.mathvista_file)
        if not mathvista_file.exists():
            raise FileNotFoundError(f"MathVista parquet file not found at {mathvista_file}")

        mathvista_records = load_mathvista_records(mathvista_file)
        print_note(f"Loaded {len(mathvista_records)} MathVista samples from {mathvista_file}")

        calibration_source = None
        calibration_records = None
        if method == "mgap":
            if args.mathvista_calibration_source == "mathvista":
                if args.mathvista_calibration_file is not None:
                    print_warning("Ignoring --mathvista-calibration-file because --mathvista-calibration-source mathvista was requested.")
                calibration_records = mathvista_records
                calibration_source = f"mathvista:{mathvista_file}"
                print_note(f"Building MathVista subspace from MathVista data: {mathvista_file}")
            else:
                calibration_file = args.mathvista_calibration_file
                if calibration_file is None:
                    if not os.path.exists(args.pope_dir):
                        if os.path.exists("./POPE/coco"):
                            args.pope_dir = "./POPE/coco"
                        else:
                            raise FileNotFoundError(f"POPE directory not found at {args.pope_dir}")
                    calibration_file = os.path.join(args.pope_dir, "coco_pope_random.json")

                if not os.path.exists(calibration_file):
                    raise FileNotFoundError(f"MathVista calibration file not found at {calibration_file}")

                calibration_source = calibration_file
                print_note(f"Building MathVista subspace from POPE calibration data: {calibration_file}")
                with open(calibration_file, 'r', encoding='utf-8') as f:
                    calibration_records = [json.loads(line) for line in f]

        if method in {"haltrapper", "code"}:
            print_note("Using CHAIR parser for candidate extraction on MathVista...")
            parser = ChairParser()
            model.parser = parser
            model.ct = ContextCDCandidates(model, parser)

        if args.log_file_path is not None:
            log_file_path = Path(args.log_file_path)
            if log_file_path.suffix != ".jsonl":
                log_file_path = log_file_path.with_suffix(".jsonl")
        else:
            ablation_suffix = ""
            if method == "mgap":
                if args.no_protection:
                    ablation_suffix += "_noprot"
                if args.no_gate:
                    ablation_suffix += "_nogate"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file_path = output_dir / f"{model.name}-mathvista-{method}{ablation_suffix}_{timestamp}.jsonl"

        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        open_mode = "w" if args.overwrite else "x"

        save_structured_file(
            {
                "args": sys.argv,
                "model": model.name,
                "benchmark": "mathvista",
                "method": method,
                "start_time": datetime.now().astimezone().isoformat(),
                "kwargs": {**kwargs, **method_kwargs},
                "mathvista_file": os.fspath(mathvista_file),
                "calibration_source": calibration_source,
            },
            log_file_path.parent / (log_file_path.stem + "-config.yaml"),
            "w",
        )

        if method == "mgap":
            subspace_manager = BiasSubspaceManager(model.model.device)
            subspace_manager.calibrate(
                model,
                calibration_records,
                args,
                n_samples=min(args.calibration_samples, len(calibration_records)),
            )
            model.subspace_manager = subspace_manager

        total = 0
        correct = 0
        print_note(f"Saving MathVista results to {log_file_path}")

        with open(log_file_path, open_mode, encoding="utf-8") as ans_file:
            for i, item in enumerate(tqdm(mathvista_records)):
                extra_info = normalize_structured_value(item.get("extra_info"))
                question_id = extra_info.get("index", i) if isinstance(extra_info, dict) else i
                prompt = extract_mathvista_prompt(item)
                ground_truth = extract_mathvista_ground_truth(item)

                image_path = None
                submit_result = None
                remove_temp_image = False
                try:
                    image_path, remove_temp_image = materialize_mathvista_image(
                        item.get("images"),
                        question_id,
                    )

                    submit_result = model.submit(
                        prompt=prompt,
                        image=image_path,
                        method=method,
                        question_id=question_id,
                        **kwargs,
                        **method_kwargs,
                    )
                finally:
                    if remove_temp_image and image_path is not None and os.path.exists(image_path):
                        os.remove(image_path)

                if len(submit_result) == 4:
                    response, _, _, _ = submit_result
                else:
                    response, _, _ = submit_result

                pred_answer = extract_mathvista_prediction(response, ground_truth)
                is_correct = mathvista_answer_is_correct(pred_answer, ground_truth)
                total += 1
                correct += int(is_correct)

                ans_file.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "prompt": prompt,
                            "text": response,
                            "pred_answer": pred_answer,
                            "gt_answer": ground_truth,
                            "correct": is_correct,
                            "model_id": model_name,
                            "task": extra_info.get("task") if isinstance(extra_info, dict) else None,
                            "question": extra_info.get("question") if isinstance(extra_info, dict) else None,
                            "ability": item.get("ability"),
                            "metadata": {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ans_file.flush()

        accuracy = correct / total if total else 0.0
        print("\n" + "=" * 40)
        print("🏆 MathVista Summary")
        print("=" * 40)
        print(f"Samples  : {total}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Results  : {log_file_path}")
        print("=" * 40)

        if method in {"haltrapper", "code"}:
            model.close_cache_table()

    else:

        remain_args_with_eval = ["--eval", args.eval] + remain_args
        benchmark, remain_args = get_eval_benchmark_from_args(remain_args_with_eval)
        assert benchmark is not None

        if benchmark.name == "amber":
            print_note("Using AMBER parser...")
            parser = AmberParser()
        else:
            print_note("Using CHAIR parser...")
            parser = ChairParser()

        model.parser = parser
        model.ct = ContextCDCandidates(model, parser)

        if args.log_file_path is not None:
            log_file_path: str = args.log_file_path
            if not log_file_path.endswith(".jsonl"):
                log_file_path += ".jsonl"
        else:

            ablation_suffix = ""
            if method == "mgap":
                if args.no_protection:
                    ablation_suffix += "_noprot"
                if args.no_gate:
                    ablation_suffix += "_nogate"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file_path = str(
                output_dir / f"{model.name}-{benchmark.name}-{method}{ablation_suffix}_{timestamp}.jsonl"
            )

        model.set_log_file_path(log_file_path, "w" if args.overwrite else "X")
        log_file_path = model.log_file_path

        save_structured_file(
            {
                "args": sys.argv,
                "model": model.name,
                "benchmark": benchmark.name,
                "method": method,
                "start_time": datetime.now().astimezone().isoformat(),
                "kwargs": {**kwargs, **method_kwargs},
            },
            log_file_path.parent / (log_file_path.stem + "-config.yaml"),
            "w",
        )

        if method == "mgap":
            subspace_manager = BiasSubspaceManager(model.model.device)
            subspace_manager.calibrate(model, benchmark, args, n_samples=args.calibration_samples)
            model.subspace_manager = subspace_manager

        model.eval(
            benchmark,
            method=method,
            **kwargs,
            **method_kwargs,
        )
