from .graber import graber as GRABER

import os
import hashlib
import json
import re
from collections import defaultdict
import torch
from torch.nn import functional as F
from tqdm import tqdm
import atexit

from typing import Generator, List, Tuple, Dict, Union, Optional, Any, TYPE_CHECKING
from playground._utils._path import PathObj
from playground._utils._colors import print_note, print_warning
from playground import LM

if TYPE_CHECKING:
    from parsers import BaseParser

class CacheBaseDB:
    def __init__(self, db_path: PathObj) -> None:
        print_note(f"⚠️ [NOTICE] Using IN-MEMORY Cache logic. No file will be created at {db_path}")
        self.db_path = db_path
        self.memory_db = {}

    def close(self): pass
    def __setitem__(self, key: str, value: str) -> None: self.memory_db[str(key)] = value
    def __getitem__(self, key: str) -> str:
        if str(key) not in self.memory_db: raise KeyError(f"Key '{key}' not found.")
        return self.memory_db[str(key)]
    def __delitem__(self, key: str) -> None:
        if str(key) in self.memory_db: del self.memory_db[str(key)]
    def __iter__(self) -> Generator[str, None, None]:
        for key in self.memory_db: yield key
    def __contains__(self, key: str): return str(key) in self.memory_db
    def items(self) -> List[Tuple[str, str]]: return list(self.memory_db.items())
    def keys(self) -> List[str]: return list(self.memory_db.keys())
    def values(self) -> List[str]: return list(self.memory_db.values())

def get_file_hash(file_path: PathObj) -> str:
    if not os.path.exists(str(file_path)): return "hash_skipped_file_not_found"
    md5_hash = hashlib.md5()
    i = 0
    with open(file_path, "rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            md5_hash.update(byte_block)
            i += 1
            if i >= 64: break
    return md5_hash.hexdigest()

class Cache:
    def __init__(self, model_name: str, subfolder_name: str, check_hash: bool = True) -> None:
        self.db = CacheBaseDB(f"./cache/{subfolder_name}/{model_name}.db")
        self.check_hash = check_hash
    def close(self): self.db.close()
    @staticmethod
    def dump_dict(value: Dict[Any, Any]) -> str: return json.dumps(value)
    @staticmethod
    def load_dict(value: str) -> Dict[Any, Any]: return json.loads(value)
    def __setitem__(self, key: PathObj, value: Dict[Any, Any]) -> None:
        value["__hash__"] = get_file_hash(key)
        self.db[str(key)] = self.dump_dict(value)
    def __getitem__(self, key: PathObj) -> Dict[Any, Any]:
        value = self.db[str(key)]
        value = self.load_dict(value)
        return value
    def __delitem__(self, key: PathObj) -> None: del self.db[str(key)]
    def __iter__(self) -> Generator[str, None, None]: return self.db.__iter__()
    def __contains__(self, key: PathObj) -> bool: return str(key) in self.db
    def items(self) -> Generator[Tuple[str, Dict[Any, Any]], None, None]:
        items = self.db.items()
        for key, value in items: yield key, self.load_dict(value)
    def keys(self) -> List[str]: return self.db.keys()
    def values(self) -> Generator[Dict[Any, Any], None, None]:
        values = self.db.values()
        for value in values: yield self.load_dict(value)

class ContextCDCandidates:
    CAPTION_PROMPT = "Please help me describe the {vis_type} in detail."
    METRIC1_DIRECTIONS = {
        "N": "top", "S": "bottom", "W": "left side", "E": "right side",
        "NW": "top left corner", "NE": "top right corner", "SW": "bottom left corner", "SE": "bottom right corner",
    }
    METRIC1_PROMPT = """
Based on this {vis_type}, please imagine what object might be in the {direction} outside the frame, and explain why. Specifically, your response should follow the following format:

Imagination: <one imaginary object here>
Reason: The {vis_type} features <briefly describe this {vis_type}, be careful to mention all objects related to your imagination>, which suggests that <your imagination here>.
    """.strip()
    METRIC2_PROMPT = " There is also"
    GREEDY_KWARGS = {"do_sample": False, "max_new_tokens": 512}
    METRIC2_KWARGS = {"do_sample": False, "max_new_tokens": 4}

    def __init__(self, model: Union[LM, str], parser: "BaseParser", check_hash: bool = True) -> None:
        if isinstance(model, LM):
            model_name = model.name
            self.model = model
        else:
            model_name = model
            self.model = None
        self.cache = Cache(model_name, parser.name, check_hash)
        self.parser = parser

    def close(self): self.cache.close()
    def in_cache(self, img_path: PathObj) -> bool: return img_path in self.cache

    def get_candidates(self, img_path: PathObj, show_progress: bool = False) -> Dict[Any, Any]:
        if str(img_path).lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv")):
            vis_type = "video"
        else:
            vis_type = "image"
        if img_path in self.cache:
            if show_progress: tqdm.write(f"Candidates for {repr(img_path)} is in database.")
            return self.cache[img_path]
        if self.model is None:
            raise KeyError(f"Candidates for {repr(img_path)} is NOT in database.")
        else:
            if show_progress: tqdm.write(f"Generating candidates for {repr(img_path)}...")
            return self.generate_candidates(img_path, vis_type, show_progress)

    def generate_metric1(self, img_path: PathObj, vis_type: str, show_progress: bool = False) -> Dict[Any, Any]:
        assert self.model is not None
        responses: Dict[str, Dict[Any, Any]] = {}
        if show_progress: it_obj = tqdm(self.METRIC1_DIRECTIONS, leave=False)
        else: it_obj = self.METRIC1_DIRECTIONS

        for key in it_obj:
            response, _, _ = self.model(
                self.METRIC1_PROMPT.format(direction=self.METRIC1_DIRECTIONS[key], vis_type=vis_type),
                img_path, use_log=False, **self.GREEDY_KWARGS,
            )
            responses[key] = {"raw": response}

        for key in self.METRIC1_DIRECTIONS:
            pattern = r"Imagination: (.*)\n?Reason: (.*?) (?:suggests that|might be|likely that|indicates that) (.*)"
            match = re.match(pattern, responses[key]["raw"])
            if match:
                responses[key]["imagination"] = match.group(1).strip()
                responses[key]["reason"] = match.group(2).strip()
                responses[key]["imagination"] += " " + match.group(3).strip()
            else:
                pattern = r"Imagination: (.*)\n?Reason: (.*)"
                match = re.match(pattern, responses[key]["raw"])
                if match:
                    responses[key]["imagination"] = match.group(1).strip()
                    responses[key]["reason"] = match.group(2).strip()
                else:
                    responses[key]["imagination"] = None
                    responses[key]["reason"] = None

        for key in self.METRIC1_DIRECTIONS:
            if responses[key]["imagination"] is not None:
                _, imagination_objs, _, _ = self.parser.extract_nouns(responses[key]["imagination"])
                responses[key]["imagination_objs"] = list(set(imagination_objs))
            else:
                responses[key]["imagination_objs"] = []
            if responses[key]["reason"] is not None:
                _, reason_objs, _, _ = self.parser.extract_nouns(responses[key]["reason"])
                responses[key]["reason_objs"] = list(set(reason_objs))
            else:
                responses[key]["reason_objs"] = []

        scores_dict = defaultdict(int)
        for key in self.METRIC1_DIRECTIONS:
            for im_obj in responses[key]["imagination_objs"]: scores_dict[im_obj] -= 1
            for rs_obj in responses[key]["reason_objs"]: scores_dict[rs_obj] += 1
        return {"scores": dict(scores_dict), "debug_info": responses}

    def generate_metric2(self, img_path: PathObj, caption: str, vis_type: str, prompt: Optional[str] = None, show_progress: bool = False) -> Dict[Any, Any]:
        assert self.model is not None
        if prompt is None: prompt = self.CAPTION_PROMPT.format(vis_type=vis_type)

        cur_response, cur_output, _ = self.model(
            prompt, img_path,
            return_dict_in_generate=True, output_attentions=True,
            append=caption + self.METRIC2_PROMPT,
            use_log=False, **self.METRIC2_KWARGS,
        )

        image_start_pos: Optional[int] = GRABER.pop("image_start_pos")
        image_end_pos: Optional[int] = GRABER.pop("image_end_pos")
        input_ids_offset: int = GRABER.pop("input_ids_offset")

        GRABER.pop("input_ids")

        if image_start_pos is None or image_end_pos is None or input_ids_offset is None:

             return {"scores": {}, "debug_info": "Missing GRABER data"}

        if not cur_output or "attentions" not in cur_output or len(cur_output["attentions"]) == 0:
            return {"scores": {}, "debug_info": "Empty generation/attentions"}

        try:
            full_sequences = cur_output.sequences
            generated_tokens_ids = full_sequences[0][input_ids_offset:]
        except (AttributeError, KeyError, IndexError):
            return {"scores": {}, "debug_info": "Failed to extract generated tokens"}

        decoded_tokens: List[str] = []
        start_poses: List[Optional[int]] = []
        end_poses: List[Optional[int]] = []
        end_pos = 0

        for token in generated_tokens_ids:
            decoded_token = self.model.tokenizer.decode(token, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            decoded_tokens.append(decoded_token)
            start_pos = caption.find(decoded_token, end_pos)
            if start_pos == -1:
                start_poses.append(None)
                end_poses.append(None)
            else:
                end_pos = start_pos + len(decoded_token)
                start_poses.append(start_pos)
                end_poses.append(end_pos)

        try:
            sel_input_attns = cur_output["attentions"][0]
            sel_input_attns = torch.stack(sel_input_attns, dim=0)

            sel_output_token = 1
            if len(cur_output["attentions"]) <= sel_output_token:
                 sel_output_attns_raw = cur_output["attentions"][-1]
            else:
                 sel_output_attns_raw = cur_output["attentions"][sel_output_token]

            sel_output_attns = [attn[0, :, -1, :] for attn in sel_output_attns_raw]
            sel_output_attns = torch.stack(sel_output_attns, dim=0)

            sel_output_image_attns = sel_output_attns[:, :, image_start_pos:image_end_pos]
            sel_output_image_attns_flat = torch.flatten(sel_output_image_attns).double()

            output_similarities = []
            auto_sel_end = sel_input_attns.shape[3] - 1
            auto_sel_start = auto_sel_end - len(decoded_tokens)

            for sel_input_token in range(auto_sel_start, auto_sel_end):
                cur_sel_input_attns = sel_input_attns[:, 0, :, sel_input_token, :]
                sel_input_image_attns = cur_sel_input_attns[:, :, image_start_pos:image_end_pos]
                image_attn_weight = sel_input_image_attns.mean()
                sel_input_image_attns_flat = torch.flatten(sel_input_image_attns).double()
                cos_sim = F.cosine_similarity(sel_input_image_attns_flat, sel_output_image_attns_flat, dim=0)
                output_similarities.append({
                    "cos_sim": float(cos_sim),
                    "image_attn_weight": float(image_attn_weight),
                })
        except Exception as e:
            return {"scores": {}, "debug_info": f"Attn Calc Error: {e}"}

        (object_tokens, object_lemmas, object_start_poses, object_end_poses) = self.parser.extract_nouns(caption)
        lemmas_ptr: List[Optional[int]] = []

        if object_start_poses:
            i = 0
            for token, s, e in zip(decoded_tokens, start_poses, end_poses):
                if s is None or e is None:
                    lemmas_ptr.append(None)
                elif object_start_poses[i] <= e:
                    lemmas_ptr.append(i)
                    i += 1
                    if i >= len(object_start_poses): break
                else:
                    lemmas_ptr.append(None)

        scores = {}
        for p, sim in zip(lemmas_ptr, output_similarities):
            if p is None: continue
            lemma = object_lemmas[p]
            if lemma in scores.keys(): continue
            scores[lemma] = sim

        return {
            "scores": scores,
            "debug_info": {
                "caption": caption,
                "response": cur_response,
                "transformers": {
                    "tokens": decoded_tokens,
                    "similarities": output_similarities,
                },
                "parser": {"tokens": object_tokens, "lemmas": object_lemmas},
                "alignment": lemmas_ptr,
            },
        }

    def generate_candidates(self, img_path: PathObj, vis_type: str, show_progress: bool = False) -> Dict[Any, Any]:
        assert self.model is not None
        return_dict = {}

        if show_progress: tqdm.write("Generating Caption...")
        _, output, _ = self.model(
            self.CAPTION_PROMPT.format(vis_type=vis_type),
            img_path, use_log=False, return_dict_in_generate=True, **self.GREEDY_KWARGS,
        )
        assert output is not None
        if self.model.name == "Qwen2-VL-7B-Instruct":
            caption = self.model.tokenizer.decode(output["sequences"][0][GRABER["input_ids_offset"] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            caption = self.model.tokenizer.decode(output["sequences"][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return_dict["caption"] = caption

        if show_progress: tqdm.write("Getting metric 1...")
        return_dict["metric1"] = self.generate_metric1(img_path, vis_type, show_progress)

        if show_progress: tqdm.write("Getting metric 2...")
        return_dict["metric2"] = self.generate_metric2(img_path, caption, vis_type, None, show_progress)

        self.cache[img_path] = return_dict
        if show_progress: tqdm.write("Done!")
        return return_dict
