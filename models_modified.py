import types
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import os
from abc import ABC
from typing import TYPE_CHECKING, Dict, Tuple, Optional
import transformers

from methods_utils.new_modeling_llama_4_37_2 import (
    new_LlamaForCausalLM_forward,
    new_LlamaModel_forward,
    new_LlamaSdpaAttention_forward,
    new_LlamaAttention_forward,
)
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaSdpaAttention,
    LlamaAttention,
    LlamaForCausalLM
)

LlamaModel.forward = new_LlamaModel_forward
LlamaForCausalLM.forward = new_LlamaForCausalLM_forward
LlamaSdpaAttention.forward = new_LlamaSdpaAttention_forward
LlamaAttention.forward = new_LlamaAttention_forward

from llava.constants import IMAGE_TOKEN_INDEX
from playground.models import LLaVA
from playground.path_table import get_path_from_table
from parsers import BaseParser
from playground._utils._colors import *
from methods_utils.cache_table import ContextCDCandidates
from methods_utils.vcd_add_noise import add_diffusion_noise

try:
    from methods_utils.graber import graber as GRABER
except ImportError:
    from methods_utils.graber import Graber
    GRABER = Graber()

if TYPE_CHECKING:
    from playground._utils._path import PathObj

class CTMixin(ABC):
    ct: ContextCDCandidates
    parser: BaseParser

    def get_candidates_from_cache(
        self,
        image: "PathObj",
        number: int,
        ee_threshold: float,
        ig_threshold: float,
        ig_strategy: str,
        show_progress: bool = False,
        random_state: Optional[int] = None,
    ) -> list[str]:
        datadict = self.ct.get_candidates(image, show_progress)
        _, caption_objs, _, _ = self.parser.extract_nouns(datadict["caption"])

        appeared = set(caption_objs)
        appeared |= set(datadict["metric1"]["scores"].keys())
        appeared |= set(datadict["metric2"]["scores"].keys())

        related = set()
        for word in appeared:
            related.add(word)
            for subword in self.parser.SAFE_WORDS[word]:
                related.add(subword)

        related_caption = set()
        for word in caption_objs:
            related_caption.add(word)
            for subword in self.parser.SAFE_WORDS[word]:
                related_caption.add(subword)

        unrelated = set(self.parser.PARSER_WORDS) - related
        unrelated = sorted(unrelated)

        candidates = {}

        for obj, score in datadict["metric2"]["scores"].items():
            score_strategy = score[ig_strategy]
            if score_strategy > ig_threshold:
                candidates[obj] = score_strategy

        EE_list = []
        for obj, score in datadict["metric1"]["scores"].items():
            if score < ee_threshold and obj not in candidates.keys():
                EE_list.append(obj)
                candidates[obj] = -1.0

        candidates = sorted(candidates, key=candidates.get, reverse=True)

        if random_state is not None:
            random_generator = random.Random(random_state)
        else:
            random_generator = random

        if number >= 1:
            if len(candidates) > number:
                candidates = candidates[:number]
            else:
                candidates = candidates + random_generator.sample(
                    unrelated, min(number - len(candidates), len(unrelated))
                )

        candidates = sorted(candidates)
        return candidates

    def preprocess_method(
        self,
        image,
        method,
        candidates_number,
        ee_threshold,
        ig_threshold,
        ig_strategy,
        question_id: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        GRABER.clear()
        if method in ("ours", "clearsight", "urs_vcd"):
            method = "mgap"

        if method == "haltrapper":
            assert image is not None
            candidates = {}
            for obj in self.get_candidates_from_cache(
                image,
                candidates_number,
                ee_threshold,
                ig_threshold,
                ig_strategy,
                random_state=question_id,
                show_progress=False,
            ):
                candidates[obj] = 0.0
            hallu_objs = candidates
        else:
            hallu_objs = None

        if method == "code":
            caption = self.ct.get_candidates(image, show_progress=False)["caption"]
        else:
            caption = None

        return hallu_objs, caption

    def close_cache_table(self):
        self.ct.close()

class LlavaModified(LLaVA, CTMixin):
    def __init__(self, size="7b", model_path: Optional[str] = None) -> None:
        from transformers.generation.utils import GenerationMixin
        from methods_utils.new_llava_llama import new_LlavaLlamaForCausalLM_generate, new_LlavaLlamaForCausalLM_forward
        from llava import LlavaLlamaForCausalLM
        from methods_utils.search_methods_4_37_2 import new_greedy_search, new_sample, new_beam_search

        GenerationMixin.greedy_search = new_greedy_search
        GenerationMixin.sample = new_sample
        GenerationMixin.beam_search = new_beam_search
        LlavaLlamaForCausalLM.generate = new_LlavaLlamaForCausalLM_generate
        LlavaLlamaForCausalLM.forward = new_LlavaLlamaForCausalLM_forward

        super().__init__("1.5", size)

        if model_path is not None:
            self.model_path = model_path

    def generate_mgap(self, input_ids, images, subspace_manager, alpha, gate_sensitivity, cos_threshold, norm_restoration, max_new_tokens=512, record_internals=False, protection_ablation=True, gate_ablation=True, **kwargs):
        output_ids = input_ids.clone()
        past_key_values = None
        first_token_record = None

        pai_alpha = kwargs.pop('pai_alpha', None)
        if pai_alpha is not None:
            GRABER["pai_alpha"] = pai_alpha

        generation_params = [
            "temperature", "do_sample", "top_p", "num_beams",
            "repetition_penalty", "max_new_tokens", "return_dict_in_generate"
        ]
        forward_kwargs = {k: v for k, v in kwargs.items() if k not in generation_params}

        model_inputs_initial = self.model.prepare_inputs_for_generation(output_ids, images=images, past_key_values=None)

        for i in range(max_new_tokens):
            if i == 0:
                model_inputs = model_inputs_initial
            else:
                model_inputs = self.model.prepare_inputs_for_generation(output_ids[:, -1:], images=None, past_key_values=past_key_values)

            with torch.no_grad():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    **forward_kwargs
                )

                past_key_values = outputs.past_key_values
                h_orig = outputs.hidden_states[-1][:, -1, :]
                logits_orig = outputs.logits[:, -1, :]

                probs = torch.softmax(logits_orig, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
                gate = torch.tanh(gate_sensitivity * entropy).unsqueeze(-1)

                bias_vector = subspace_manager.project_vector(h_orig)

                if record_internals and i == 0:
                    first_token_record = {
                        'h_orig': h_orig.cpu().numpy(),
                        'logits_orig': logits_orig.cpu().numpy(),
                        'probs': probs.cpu().numpy(),
                        'entropy': entropy.cpu().item(),
                        'gate': gate.cpu().numpy(),
                        'bias_vector': bias_vector.cpu().numpy()
                    }

                cos_sim = F.cosine_similarity(h_orig.float(), bias_vector.float(), dim=-1).unsqueeze(-1)
                protection = torch.where(
                    cos_sim > cos_threshold,
                    (1.0 - cos_sim) / (1.0 - cos_threshold),
                    torch.tensor(1.0, device=cos_sim.device)
                )
                protection = torch.clamp(protection, min=0.0)

                if not protection_ablation:
                    protection = torch.ones_like(protection)
                if not gate_ablation:
                    gate = torch.ones_like(gate)

                effective_alpha = alpha * gate * protection
                h_final = h_orig - effective_alpha * bias_vector

                if norm_restoration:
                    norm_orig = torch.norm(h_orig, dim=-1, keepdim=True)
                    norm_final = torch.norm(h_final, dim=-1, keepdim=True)
                    h_final = h_final / (norm_final + 1e-6) * norm_orig

                h_final = h_final.to(self.model.lm_head.weight.dtype)
                final_logits = self.model.lm_head(h_final)

                next_token = torch.argmax(final_logits, dim=-1).unsqueeze(-1)
                output_ids = torch.cat([output_ids, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        if pai_alpha is not None and "pai_alpha" in GRABER:
            del GRABER["pai_alpha"]

        return output_ids, first_token_record

    def new_eval_model_pretrained(self, args, disable_conv_mode_warning=False, **kwargs):
        import torch
        from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER)
        from llava.conversation import conv_templates
        from llava.utils import disable_torch_init
        from llava.mm_utils import (process_images, tokenizer_image_token)
        from methods_utils.vcd_add_noise import add_diffusion_noise
        import re

        disable_torch_init()

        qs = args.query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if args.image_file is not None:
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower(): conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower(): conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower(): conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower(): conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower(): conv_mode = "mpt"
        else: conv_mode = "llava_v0"

        if conv_mode == "llava_v0" and args.conv_mode is None and not disable_conv_mode_warning: pass
        if args.conv_mode is not None and conv_mode != args.conv_mode: pass
        else: args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if args.image_file is None:
            images_tensor = None
            image_sizes = None
        else:
            image_files = self.image_parser(args)
            images = self.load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images, self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)

        if args.append is not None:
            input_ids_origin = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            GRABER["input_ids_offset"] = len(input_ids_origin[0])
            prompt += " " + args.append

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        GRABER["input_ids"] = input_ids
        if args.append is None:
            GRABER["input_ids_offset"] = len(input_ids[0])

        if input_ids is not None:

            matches = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)
            if len(matches[1]) > 0:
                start_idx = matches[1][0].item()

                image_len = 576
                GRABER["image_start_pos"] = start_idx
                GRABER["image_end_pos"] = start_idx + image_len
            else:

                if "image_start_pos" in GRABER: del GRABER["image_start_pos"]
                if "image_end_pos" in GRABER: del GRABER["image_end_pos"]

        method = args.method
        output = None
        input_ids_cd = None
        images_cd = None
        input_scaling = None
        input_scaling_cd = None
        use_cd = False

        if method == "icd":
            use_cd = True
            icd_prompt = "You are a confused objects detector to provide a fuzzy overview or impression of the image. " + qs
            if args.image_file is not None:
                 if IMAGE_PLACEHOLDER in icd_prompt:
                    if self.model.config.mm_use_im_start_end:
                        icd_prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, icd_prompt)
                    else:
                        icd_prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, icd_prompt)
                 elif DEFAULT_IMAGE_TOKEN not in icd_prompt:
                    if self.model.config.mm_use_im_start_end:
                        icd_prompt = image_token_se + "\n" + icd_prompt
                    else:
                        icd_prompt = DEFAULT_IMAGE_TOKEN + "\n" + icd_prompt

            conv_cd = conv_templates[args.conv_mode].copy()
            conv_cd.append_message(conv_cd.roles[0], icd_prompt)
            conv_cd.append_message(conv_cd.roles[1], None)
            prompt_cd = conv_cd.get_prompt()
            input_ids_cd = tokenizer_image_token(prompt_cd, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            images_cd = images_tensor

        elif method == "vcd":
            use_cd = True
            input_ids_cd = input_ids.clone()
            if images_tensor is not None:
                images_cd = add_diffusion_noise(images_tensor, args.noise_step)
            else:
                images_cd = None

        elif method == "pai":
            use_cd = True
            if "pai_alpha" in kwargs:
                GRABER["pai_alpha"] = kwargs["pai_alpha"]

        if method == "mgap":
            subspace_manager = kwargs.pop('subspace_manager', getattr(self, 'subspace_manager', None))
            alpha = kwargs.pop('alpha', 1.0)
            gate_sensitivity = kwargs.pop('gate_sensitivity', 1.6)
            cos_threshold = kwargs.pop('cos_threshold', 0.15)
            norm_restoration = kwargs.pop('norm_restoration', True)
            record_internals = kwargs.pop('record_internals', False)
            protection_ablation = kwargs.pop('protection_ablation', True)
            gate_ablation = kwargs.pop('gate_ablation', True)
            _ = kwargs.pop('subspace_dim', None)

            if subspace_manager is None:
                pass
            else:
                with torch.inference_mode():
                    output_ids, first_token_record = self.generate_mgap(
                        input_ids=input_ids,
                        images=images_tensor,
                        subspace_manager=subspace_manager,
                        alpha=alpha,
                        gate_sensitivity=gate_sensitivity,
                        cos_threshold=cos_threshold,
                        norm_restoration=norm_restoration,
                        record_internals=record_internals,
                        protection_ablation=protection_ablation,
                        gate_ablation=gate_ablation,
                        **kwargs
                    )
                    output = output_ids

                    if record_internals and first_token_record is not None:
                        GRABER["first_token_record"] = first_token_record

        if output is None:

            excluded_params = [
                "temperature", "do_sample", "top_p", "num_beams",
                "repetition_penalty", "max_new_tokens", "return_dict_in_generate",
                "pai_alpha", "subspace_manager", "subspace_dim", "alpha",
                "gate_sensitivity", "cos_threshold", "norm_restoration", "record_internals",
                "protection_ablation", "gate_ablation", "use_cd", "cd_type"
            ]
            gen_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_params}
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids,
                    input_ids_cd=input_ids_cd,
                    images=images_tensor,
                    images_cd=images_cd,
                    image_sizes=image_sizes,
                    input_scaling=input_scaling,
                    input_scaling_cd=input_scaling_cd,
                    cd_type=args.cd_type,
                    use_cd=use_cd,
                    **gen_kwargs
                )

        if "return_dict_in_generate" in kwargs.keys() and kwargs["return_dict_in_generate"]:
            output_ids = output["sequences"]
        else:
            output_ids = output

        if input_ids is not None and output_ids.shape[1] > input_ids.shape[1]:
            output_ids = output_ids[:, input_ids.shape[1]:]

        response = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        return response, output

    def submit(
        self,
        prompt,
        image=None,
        question_id=None,
        method=None,
        cd_type=None,
        noise_step=None,
        repeat=1,
        repeat_mode="continuous",
        pai_alpha=None,
        append=None,
        sep=" ",
        candidates_number=None,
        ee_threshold=None,
        ig_threshold=None,
        ig_strategy="cos_sim",
        **kwargs,
    ):
        hallu_objs, caption = self.preprocess_method(
            image,
            method,
            candidates_number,
            ee_threshold,
            ig_threshold,
            ig_strategy,
            question_id,
        )

        if sep != " ": raise NotImplementedError()
        if method is None: method = "baseline"
        if hallu_objs is None and method == "haltrapper": hallu_objs = []

        args = type(
            "Args",
            (),
            {
                "model_path": self.model_path,
                "model_base": None,
                "model_name": self.get_model_name_from_path(self.model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": image,
                "sep": ",",
                "noise_step": noise_step,
                "hallu_objs": hallu_objs,
                "repeat": repeat,
                "repeat_mode": repeat_mode,
                "pai_alpha": pai_alpha,
                "method": method,
                "caption": caption,
                "cd_type": cd_type,
                "append": append,
            },
        )()

        if pai_alpha is not None:
            kwargs['pai_alpha'] = pai_alpha

        response, output = self.new_eval_model_pretrained(args, **kwargs)

        model_logs = {}
        if hallu_objs: model_logs["candidates"] = hallu_objs
        if caption: model_logs["caption"] = caption

        first_token_record = None
        if "first_token_record" in GRABER:
            first_token_record = GRABER.pop("first_token_record")

        if kwargs.get("record_internals", False):
            return response, output, model_logs if model_logs else None, first_token_record

        return response, output, model_logs if model_logs else None
