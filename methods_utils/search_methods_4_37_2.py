from transformers.generation.utils import *
from .prepare_cd import prepare_kwargs_for_cd
from transformers import __version__ as transformers_version
import torch.nn.functional as F

assert transformers_version == "4.37.2"

def new_greedy_search(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    (
        input_ids,
        _,
        model_kwargs,
        model_kwargs_cd,
        use_cd,
        cd_type,
    ) = prepare_kwargs_for_cd(input_ids, model_kwargs)

    this_peer_finished = False
    while True:
        if synced_gpus:

            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)

            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)

            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            input_scaling=model_kwargs["input_scaling"],
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        if use_cd:

            model_inputs_cd = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs_cd
            )
            outputs_cd = self(
                **model_inputs_cd,
                input_scaling=model_kwargs_cd["input_scaling"],
                return_dict=True,
                output_attentions=(
                    output_attentions
                    if output_attentions is not None
                    else self.generation_config.output_attentions
                ),
                output_hidden_states=(
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.generation_config.output_hidden_states
                ),
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if cd_type == "code":
                from methods_utils.code_dynamic_cd import code_cd

                cd_logits = code_cd(
                    model_kwargs, next_token_logits, next_token_logits_cd
                )

            else:

                cd_alpha = (
                    model_kwargs.get("cd_alpha")
                    if model_kwargs.get("cd_alpha") is not None
                    else 0.5
                )
                cd_beta = (
                    model_kwargs.get("cd_beta")
                    if model_kwargs.get("cd_beta") is not None
                    else 0.1
                )
                cd_alpha_aug = (
                    model_kwargs.get("cd_alpha_aug")
                    if model_kwargs.get("cd_alpha_aug") is not None
                    else cd_alpha
                )

                cutoff = (
                    torch.log(torch.tensor(cd_beta))
                    + next_token_logits.max(dim=-1, keepdim=True).values
                )

                if cd_type == "contrastive":
                    diffs = (
                        1 + cd_alpha
                    ) * next_token_logits - cd_alpha * next_token_logits_cd

                elif cd_type == "augmentive":
                    diffs = next_token_logits + cd_alpha_aug * next_token_logits_cd

                else:
                    raise ValueError(f"Unknown cd_type={cd_type}.")

                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_tokens_scores = logits_processor(input_ids, cd_logits)
        else:

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

def new_sample(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = (
        logits_warper if logits_warper is not None else LogitsProcessorList()
    )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    (
        input_ids,
        _,
        model_kwargs,
        model_kwargs_cd,
        use_cd,
        cd_type,
    ) = prepare_kwargs_for_cd(input_ids, model_kwargs)

    this_peer_finished = False

    while True:
        if synced_gpus:

            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)

            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)

            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            input_scaling=model_kwargs["input_scaling"],
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        if use_cd:

            model_inputs_cd = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs_cd
            )
            outputs_cd = self(
                **model_inputs_cd,
                input_scaling=model_kwargs_cd["input_scaling"],
                return_dict=True,
                output_attentions=(
                    output_attentions
                    if output_attentions is not None
                    else self.generation_config.output_attentions
                ),
                output_hidden_states=(
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.generation_config.output_hidden_states
                ),
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if cd_type == "code":
                from methods_utils.code_dynamic_cd import code_cd

                cd_logits = code_cd(
                    model_kwargs, next_token_logits, next_token_logits_cd
                )

            else:

                cd_alpha = (
                    model_kwargs.get("cd_alpha")
                    if model_kwargs.get("cd_alpha") is not None
                    else 0.5
                )
                cd_beta = (
                    model_kwargs.get("cd_beta")
                    if model_kwargs.get("cd_beta") is not None
                    else 0.1
                )
                cd_alpha_aug = (
                    model_kwargs.get("cd_alpha_aug")
                    if model_kwargs.get("cd_alpha_aug") is not None
                    else cd_alpha
                )

                cutoff = (
                    torch.log(torch.tensor(cd_beta))
                    + next_token_logits.max(dim=-1, keepdim=True).values
                )

                if cd_type == "contrastive":
                    diffs = (
                        1 + cd_alpha
                    ) * next_token_logits - cd_alpha * next_token_logits_cd

                elif cd_type == "augmentive":
                    diffs = next_token_logits + cd_alpha_aug * next_token_logits_cd

                else:
                    raise ValueError(f"Unknown cd_type={cd_type}.")

                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_token_scores = logits_processor(input_ids, cd_logits)
        else:

            next_token_scores = logits_processor(input_ids, next_token_logits)

        next_token_scores = logits_warper(input_ids, next_token_scores)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

def new_beam_search(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:

    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn(
            "You don't have defined any stopping_criteria, this will likely loop forever",
            UserWarning,
        )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False

    (
        input_ids,
        _,
        model_kwargs,
        model_kwargs_cd,
        use_cd,
        cd_type,
    ) = prepare_kwargs_for_cd(input_ids, model_kwargs)

    decoder_prompt_len = input_ids.shape[-1]
    while True:
        if synced_gpus:

            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)

            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)

            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            input_scaling=model_kwargs["input_scaling"],
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue

        next_token_logits = outputs.logits[:, -1, :]

        if use_cd:

            model_inputs_cd = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs_cd
            )
            outputs_cd = self(
                **model_inputs_cd,
                input_scaling=model_kwargs_cd["input_scaling"],
                return_dict=True,
                output_attentions=(
                    output_attentions
                    if output_attentions is not None
                    else self.generation_config.output_attentions
                ),
                output_hidden_states=(
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.generation_config.output_hidden_states
                ),
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            if cd_type == "code":
                from methods_utils.code_dynamic_cd import code_cd

                cd_logits = code_cd(
                    model_kwargs, next_token_logits, next_token_logits_cd
                )

            else:

                cd_alpha = (
                    model_kwargs.get("cd_alpha")
                    if model_kwargs.get("cd_alpha") is not None
                    else 0.5
                )
                cd_beta = (
                    model_kwargs.get("cd_beta")
                    if model_kwargs.get("cd_beta") is not None
                    else 0.1
                )
                cd_alpha_aug = (
                    model_kwargs.get("cd_alpha_aug")
                    if model_kwargs.get("cd_alpha_aug") is not None
                    else cd_alpha
                )

                cutoff = (
                    torch.log(torch.tensor(cd_beta))
                    + next_token_logits.max(dim=-1, keepdim=True).values
                )

                if cd_type == "contrastive":
                    diffs = (
                        1 + cd_alpha
                    ) * next_token_logits - cd_alpha * next_token_logits_cd

                elif cd_type == "augmentive":
                    diffs = next_token_logits + cd_alpha_aug * next_token_logits_cd

                else:
                    raise ValueError(f"Unknown cd_type={cd_type}.")

                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_token_scores = nn.functional.log_softmax(
                cd_logits, dim=-1
            )
        else:
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs_cd["past_key_values"] is not None:
                model_kwargs_cd["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs_cd["past_key_values"], beam_idx
                )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]
