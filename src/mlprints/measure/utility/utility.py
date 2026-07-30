"""LightEval utility benchmarks for a loaded model and tokenizer."""

import json
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc

from mlprints.common.evals import resolve_task_params
from mlprints.common.utils import (
    get_context_length_from_model,
    get_context_length_from_tokenizer,
    get_model_id,
)
from mlprints.inference import run_inference, run_inference_logprobs
from mlprints.measure.utility.custom import (
    configure_chat_metric,
    gsm8k_postprocess,
    is_gsm8k_task,
)

class SaveDetailsEvaluationTracker(EvaluationTracker):
    """Standard ``EvaluationTracker`` with an optional ``details_path_template``.

    Upstream exposes ``results_path_template`` for the results JSON but hardcodes
    the details path to ``{output_dir}/details/{model}/{date_id}/``. Since our
    ``tracker_output_dir`` is already scoped per (model, benchmark, run), we want
    to flatten that; this subclass adds the missing knob. We override only
    ``_get_details_sub_folder`` because upstream's ``save_details`` already
    dispatches through it.
    """

    def __init__(
        self,
        *args,
        details_path_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.details_path_template = details_path_template

    def _get_details_sub_folder(self, date_id: str):
        if not self.details_path_template:
            return super()._get_details_sub_folder(date_id)

        base = Path(
            self.details_path_template.format(
                output_dir=self.output_dir,
                model=self.general_config_logger.model_name.strip("/"),
                date_id=date_id,
            )
        )
        if date_id in ("first", "last"):
            if not self.fs.exists(base.parent):
                raise FileNotFoundError(f"Details directory {base.parent} does not exist")
            folders = [f["name"] for f in self.fs.listdir(base.parent) if f["type"] == "directory"]
            if not folders:
                raise FileNotFoundError(f"No timestamp folders found in {base.parent}")
            date_id = max(folders) if date_id == "last" else min(folders)
            base = base.parent / date_id
        return base


class MLprintsLightevalModelConfig(ModelConfig):
    model_name: str
    batch_size: Optional[int] = 1
    max_length: Optional[int] = None
    add_special_tokens: bool = False
    override_chat_template: Optional[bool] = None
    system_prompt: Optional[str] = None
    generation_parameters: GenerationParameters = GenerationParameters()
    debug_dump_first_n: int = 0
    debug_dump_path: Optional[str] = None
    loglikelihood_microbatch_size: Optional[int] = None


class MLprintsLightevalModel(LightevalModel):
    """Lighteval wrapper model for utility measurements in the mlprints library."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: MLprintsLightevalModelConfig,
    ):
        self.config = config
        self.model = model
        self._tokenizer = tokenizer
        self._add_special_tokens = bool(config.add_special_tokens)
        if config.max_length:
            self._max_length = config.max_length
        else:
            try:
                self._max_length = get_context_length_from_model(model)
            except Exception:
                self._max_length = get_context_length_from_tokenizer(tokenizer)
        self.prompt_manager = PromptManager(
            use_chat_template=(
                config.override_chat_template
                if config.override_chat_template is not None
                else bool(tokenizer.chat_template)
            ),
            tokenizer=tokenizer,
            system_prompt=config.system_prompt,
        )
        # lighteval Pipeline calls ``model._cache._init_registry(registry)`` once at
        # setup; we don't cache, so this is a minimal stub that satisfies the hook.
        self._cache = types.SimpleNamespace(_init_registry=lambda _registry: None)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return int(self._max_length)

    def greedy_until(self, docs: List[Doc]) -> List[ModelResponse]:
        responses = [None] * len(docs)
        base_gen_kwargs = self.config.generation_parameters.to_transformers_dict()
        base_stop_strings = list(base_gen_kwargs.pop("stop_strings", []) or [])
        base_gen_kwargs.pop("output_scores", None)
        base_gen_kwargs.pop("return_dict_in_generate", None)
        if base_gen_kwargs.pop("cache_implementation", None) is not None:
            raise ValueError(
                "cache_implementation is not supported by the in-memory evaluation adapter"
            )
        debug_limit = max(0, self.config.debug_dump_first_n)
        debug_path = Path(self.config.debug_dump_path) if self.config.debug_dump_path else None
        debug_written = 0
        if debug_path:
            debug_path.parent.mkdir(parents=True, exist_ok=True)

        batch_size = max(1, int(self.config.batch_size or 1))
        doc_data = []
        for doc in docs:
            prompt = self.prompt_manager.prepare_prompt(doc)
            input_ids = self.tok_encode(prompt)

            doc_gen = dict(base_gen_kwargs)
            doc_generation_size = doc.generation_size
            if doc_generation_size is not None and doc_generation_size < 0:
                doc_generation_size = None
            max_new_tokens = doc_generation_size if doc_generation_size is not None else doc_gen.get("max_new_tokens")
            if max_new_tokens is not None:
                doc_gen["max_new_tokens"] = max_new_tokens
            num_return_sequences = doc.num_samples or doc_gen.get("num_return_sequences") or 1

            do_sample = bool(
                (doc_gen.get("temperature") or 0) > 0
                or doc_gen.get("top_p") is not None
                or doc_gen.get("top_k") is not None
                or doc_gen.get("min_p") is not None
            )
            if not do_sample:
                doc_gen["temperature"] = None
                doc_gen["top_p"] = None
                doc_gen["top_k"] = None

            stop_sequences = list(doc.stop_sequences or [])
            for stop in base_stop_strings:
                if stop and stop not in stop_sequences:
                    stop_sequences.append(stop)

            gen_config_key = (
                doc_gen.get("max_new_tokens"),
                do_sample,
                doc_gen.get("temperature"),
                doc_gen.get("top_p"),
                doc_gen.get("top_k"),
                num_return_sequences,
                tuple(sorted(s for s in stop_sequences if s)),
                tuple(
                    sorted(
                        (key, repr(value))
                        for key, value in doc_gen.items()
                        if key
                        not in {"max_new_tokens", "temperature", "top_p", "top_k"}
                    )
                ),
            )

            doc_data.append({
                "doc": doc,
                "prompt": prompt,
                "input_ids": input_ids,
                "gen_config_key": gen_config_key,
                "gen_kwargs": doc_gen,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "stop_sequences": stop_sequences,
            })

        groups = defaultdict(list)
        for idx, data in enumerate(doc_data):
            groups[data["gen_config_key"]].append((idx, data))

        pbar = tqdm(total=len(docs), disable=self.disable_tqdm, desc="greedy_until")

        for group_items in groups.values():
            for batch_start in range(0, len(group_items), batch_size):
                batch_items = group_items[batch_start:batch_start + batch_size]
                batch_prompts = [d["prompt"] for _, d in batch_items]
                batch_data = [d for _, d in batch_items]
                first_data = batch_data[0]
                # post-trim below is still required because StopStringCriteria
                # halts *after* the stop string appears in the decoded text, so
                # the stop substring typically remains in the output.
                gen_overrides = {}
                if first_data["stop_sequences"]:
                    gen_overrides["stop_strings"] = list(first_data["stop_sequences"])
                gen_overrides.update(
                    {
                        key: value
                        for key, value in first_data["gen_kwargs"].items()
                        if key
                        not in {"max_new_tokens", "temperature", "top_p", "top_k"}
                    }
                )
                batch_outputs = run_inference(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt_or_messages=batch_prompts,
                    apply_chat_template=False,
                    max_new_tokens=first_data["gen_kwargs"].get("max_new_tokens"),
                    do_sample=first_data["do_sample"],
                    temperature=first_data["gen_kwargs"].get("temperature"),
                    top_p=first_data["gen_kwargs"].get("top_p"),
                    top_k=first_data["gen_kwargs"].get("top_k"),
                    num_return_sequences=first_data["num_return_sequences"],
                    **gen_overrides,
                )

                num_seqs = first_data["num_return_sequences"]
                for batch_idx, (doc_idx, data) in enumerate(batch_items):
                    doc_outputs = [
                        batch_outputs[batch_idx * num_seqs + seq_idx]
                        for seq_idx in range(num_seqs)
                    ]
                    trimmed = []
                    for text in doc_outputs:
                        stop_positions = [
                            text.find(stop)
                            for stop in data["stop_sequences"]
                            if stop and stop in text
                        ]
                        trimmed.append(
                            text[: min(stop_positions)]
                            if stop_positions
                            else text
                        )

                    task_name = data["doc"].task_name
                    if is_gsm8k_task(task_name):
                        trimmed = [gsm8k_postprocess(t) for t in trimmed]

                    if debug_path and debug_written < debug_limit:
                        rec = {
                            "prompt": data["prompt"],
                            "stop_sequences": data["stop_sequences"],
                            "outputs_raw": doc_outputs,
                            "outputs_after_stop": trimmed,
                            "choices": data["doc"].choices,
                            "max_new_tokens_used": data["gen_kwargs"].get("max_new_tokens"),
                            "task_generation_size": data["doc"].generation_size,
                            "gold_index": data["doc"].gold_index,
                        }
                        with debug_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        debug_written += 1

                    responses[doc_idx] = (
                        ModelResponse(
                            input=data["prompt"],
                            input_tokens=data["input_ids"],
                            text=trimmed,
                            output_tokens=[self.tokenizer.encode(t, add_special_tokens=False) for t in trimmed],
                        )
                    )
                    pbar.update(1)

        pbar.close()

        missing = sum(1 for r in responses if r is None)
        if missing:
            raise RuntimeError(f"greedy_until produced {missing} missing responses (bug in batching logic).")

        return [r for r in responses if r is not None]

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        if not docs:
            return []
        if not self.prompt_manager.use_chat_template:
            raise ValueError("Loglikelihood evaluation requires a chat template.")

        responses = [ModelResponse() for _ in docs]
        conversations = []
        response_indices = []
        for doc_idx, doc in enumerate(docs):
            prompt = self.prompt_manager.prepare_prompt(doc)
            for choice in doc.choices:
                conversations.append(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": choice},
                    ]
                )
                response_indices.append(doc_idx)

        microbatch_size = self.config.loglikelihood_microbatch_size or 64
        for start in tqdm(
            range(0, len(conversations), microbatch_size),
            disable=self.disable_tqdm,
            desc="loglikelihood",
        ):
            batch = conversations[start : start + microbatch_size]
            results = run_inference_logprobs(
                self.model,
                self.tokenizer,
                batch,
                chat_template=(
                    "{% for message in messages %}"
                    "{{ message['content'] }}"
                    "{% endfor %}"
                ),
                return_metadata=True,
                extract_top_k=1,
            )

            for doc_idx, result in zip(
                response_indices[start : start + microbatch_size],
                results,
            ):
                response = responses[doc_idx]
                target_ids = result["target_ids"]
                if not response.input_tokens:
                    response.input_tokens = result["input_ids"][
                        : result["assistant_start_idx"]
                    ].tolist()
                response.output_tokens.append(target_ids.tolist())
                response.logprobs.append(
                    float(result["token_logprobs"].sum().item())
                )
                response.argmax_logits_eq_gold.append(
                    bool(torch.equal(result["topk_indices"][:, 0], target_ids))
                )
        return responses

    def loglikelihood_rolling(self, docs: List[Doc]) -> List[ModelResponse]:
        raise NotImplementedError("Rolling loglikelihood not supported via the mlprints lighteval wrapper.")


def evaluate_model(
    model: Any,
    tokenizer: Any,
    eval_benchmark_name: str,
    *,
    benchmark_config: Dict[str, Any],
    generation_params: Dict[str, Any],
    tracker_output_dir: Path,
    batch_size: Optional[int] = None,
    tasks: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run Lighteval; returns ``(final_dict, resolved_tasks)``."""
    config = benchmark_config.copy()

    input_tasks = tasks or eval_benchmark_name
    task_overrides = config.pop("task_overrides", {})

    default_n_shots = config.pop("n_shots", None)
    if default_n_shots is not None:
        entries = [e.strip() for e in input_tasks.split(",") if e.strip()]
        input_tasks = ",".join(
            e if "|" in e else f"{e}|{int(default_n_shots)}"
            for e in entries
        )

    execution_plan = resolve_task_params(
        input_tasks,
        task_overrides=task_overrides,
        default_max_samples=config.get("max_samples"),
        default_batch_size=batch_size,
        default_generation_size=config.get("generation_size"),
    )

    if len(execution_plan) > 1:
        all_results = {"results": {}, "groups": []}
        all_tasks = []

        for idx, group in enumerate(execution_plan):
            group_tracker_dir = tracker_output_dir / f"group_{idx:02d}"

            group_benchmark_config = config.copy()
            for key in ("max_samples", "generation_size"):
                if group[key] is None:
                    group_benchmark_config.pop(key, None)
                else:
                    group_benchmark_config[key] = group[key]

            group_results, _ = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                eval_benchmark_name=eval_benchmark_name,
                batch_size=group["batch_size"],
                benchmark_config=group_benchmark_config,
                generation_params=generation_params,
                tracker_output_dir=group_tracker_dir,
                tasks=group["tasks"],
            )

            all_results["results"].update(group_results.get("results", {}))
            all_results["groups"].append(
                {"tasks": group["tasks"], "results": group_results}
            )
            all_tasks.append(group["tasks"])

        return all_results, ",".join(all_tasks)

    group = execution_plan[0]
    resolved_tasks = group["tasks"]
    for key in ("max_samples", "generation_size"):
        if group[key] is None:
            config.pop(key, None)
        else:
            config[key] = group[key]
    batch_size = group["batch_size"]

    merged_gen_params = generation_params | config.get("generation_params", {})
    override_chat_template = merged_gen_params.pop("apply_chat_template", None)
    do_sample = merged_gen_params.pop("do_sample", None)
    unknown_gen_params = set(merged_gen_params) - set(GenerationParameters.model_fields)
    if unknown_gen_params:
        raise ValueError(
            f"Unsupported generation parameters: {sorted(unknown_gen_params)}"
        )
    if do_sample is not None and "temperature" not in merged_gen_params:
        merged_gen_params["temperature"] = 0.7 if do_sample else 0
    generation_parameters = GenerationParameters(**merged_gen_params)

    tracker_output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_tracker = SaveDetailsEvaluationTracker(
        output_dir=str(tracker_output_dir),
        results_path_template="{output_dir}",
        details_path_template="{output_dir}/details",
        save_details=True,
        push_to_hub=False,
    )

    model_name = get_model_id(model, short=True)
    debug_first_n = config.get("debug_dump_first_n", 0)
    debug_path = (
        str(tracker_output_dir / "debug_generations.jsonl") if debug_first_n > 0 else None
    )

    model_config = MLprintsLightevalModelConfig(
        model_name=model_name,
        batch_size=batch_size,
        max_length=config.get("max_length"),
        add_special_tokens=config.get("add_special_tokens", False),
        override_chat_template=override_chat_template,
        system_prompt=config.get("system_prompt"),
        generation_parameters=generation_parameters,
        loglikelihood_microbatch_size=config.get("loglikelihood_microbatch_size"),
        debug_dump_first_n=debug_first_n,
        debug_dump_path=debug_path,
    )
    wrapped_model = MLprintsLightevalModel(
        model=model,
        tokenizer=tokenizer,
        config=model_config,
    )

    pipeline_parameters = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=config.get("max_samples"),
    )

    pipeline = Pipeline(
        tasks=resolved_tasks,
        pipeline_parameters=pipeline_parameters,
        evaluation_tracker=evaluation_tracker,
        model=wrapped_model,
    )

    configure_chat_metric(eval_benchmark_name, config, pipeline)

    gen_size = config.get("generation_size")
    if gen_size is not None:
        for task_name, task in pipeline.tasks_dict.items():
            old_size = task.generation_size
            task.generation_size = gen_size
            print(
                f"Overrode generation_size for task {task_name}: "
                f"{old_size} -> {gen_size}"
            )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    final_dict = pipeline.get_results()

    return final_dict, resolved_tasks
