"""Benchmark-specific LightEval customizations."""

import logging
import re
from typing import Any, Dict, Iterable, Tuple

from aenum import extend_enum
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import ExactMatches, SampleLevelComputation
from lighteval.metrics.normalizations import harness_triviaqa_normalizer
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import Pipeline
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


CHAT_METRIC_ENUM_NAME = "CHAT_EXACT_MATCH"
CHAT_METRIC_DEFAULT_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "mmlu": ("mmlu:",),
    "hellaswag": ("hellaswag",),
}
_LETTER_PATTERNS = (
    re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Za-z])", re.IGNORECASE),
    re.compile(r"\b([A-Za-z])(?:[\)\].]|$)"),
)
_STRIP_CHAT_INTRO_PATTERN = re.compile(
    r"^(answer|final answer|response)\s*[:\-]\s*",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")
_GSM8K_MARKER_PATTERN = re.compile(r"####\s*(-?[\d,\.]+)")


def gsm8k_postprocess(text: str) -> str:
    """Append ``#### <number>`` when missing, using the last number if present."""
    if _GSM8K_MARKER_PATTERN.search(text):
        return text
    numbers = _NUMBER_PATTERN.findall(text)
    if not numbers:
        return text
    final_num = numbers[-1].replace(",", "")
    return text.rstrip() + f"\n#### {final_num}"


def is_gsm8k_task(task_name: str) -> bool:
    """Check whether a task name indicates a GSM8K task."""
    return bool(task_name) and "gsm8k" in task_name.lower()


def mean_or_zero(values: Iterable[float]) -> float:
    """Arithmetic mean of ``values`` (empty → 0.0)."""
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def configure_triviaqa_metric(pipeline: Pipeline) -> None:
    """Normalize TriviaQA predictions the same way as its gold aliases."""
    triviaqa_metric = SampleLevelMetric(
        metric_name="em",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=ExactMatches(
            normalize_gold=harness_triviaqa_normalizer,
            normalize_pred=harness_triviaqa_normalizer,
            strip_strings=True,
        ),
        corpus_level_fn=mean_or_zero,
    )

    for task_name, task in pipeline.tasks_dict.items():
        base_name = task_name.split("|", 1)[0]
        if base_name != "triviaqa":
            continue

        new_metrics = []
        replaced = False
        for metric in task.metrics:
            if (
                metric.metric_name == "em"
                and metric.category == SamplingMethod.GENERATIVE
            ):
                new_metrics.append(triviaqa_metric)
                replaced = True
            else:
                new_metrics.append(metric)
        if replaced:
            task.metrics = tuple(new_metrics)
            logger.info("Applied TriviaQA metric normalization to task %s", task_name)


def configure_chat_metric(
    eval_benchmark_name: str,
    benchmark_config: Dict[str, Any],
    pipeline: Pipeline,
) -> None:
    """Replace generative exact match with chat-aware exact match when configured."""
    chat_cfg = benchmark_config.get("chat_metric")
    default_prefixes = CHAT_METRIC_DEFAULT_PREFIXES.get(
        eval_benchmark_name,
        tuple(),
    )
    if chat_cfg is False:
        return
    if isinstance(chat_cfg, dict) and not chat_cfg.get("enabled", True):
        return
    if isinstance(chat_cfg, dict):
        prefixes = tuple(chat_cfg.get("task_prefixes") or ()) or default_prefixes
    else:
        prefixes = default_prefixes
    if not prefixes:
        return

    def ensure_chat_metric_registered():
        if hasattr(Metrics, CHAT_METRIC_ENUM_NAME):
            return getattr(Metrics, CHAT_METRIC_ENUM_NAME)

        def extract_chat_answer(doc: Doc, model_response: ModelResponse) -> str:
            text = ""
            for attr in ("final_text", "text_post_processed", "text"):
                seq = getattr(model_response, attr, None)
                if seq:
                    text = seq[0]
                    break
            text = text.strip()
            if not text:
                return text

            choices = getattr(doc, "choices", None) or []
            letter_lookup = {
                choice.strip().upper(): choice
                for choice in choices
                if (
                    isinstance(choice, str)
                    and len(choice.strip()) == 1
                    and choice.strip().isalpha()
                )
            }
            if letter_lookup:
                for pattern in _LETTER_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        letter = match.group(1).upper()
                        if letter in letter_lookup:
                            return letter_lookup[letter]
                for char in text:
                    upper = char.upper()
                    if upper in letter_lookup:
                        return letter_lookup[upper]

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                text = lines[-1]
            return _STRIP_CHAT_INTRO_PATTERN.sub("", text).strip()

        class ChatExactMatchComputation(SampleLevelComputation):
            """Generative exact match for chat-formatted answers."""

            def compute(
                self,
                doc: Doc,
                model_response: ModelResponse,
                **kwargs,
            ) -> float:
                golds = doc.get_golds() if hasattr(doc, "get_golds") else []
                predicted = extract_chat_answer(doc, model_response)
                return 1.0 if any(predicted == gold for gold in golds) else 0.0

        chat_metric = SampleLevelMetric(
            metric_name="chat_exact_match",
            higher_is_better=True,
            category=SamplingMethod.GENERATIVE,
            sample_level_fn=ChatExactMatchComputation(),
            corpus_level_fn=mean_or_zero,
        )
        extend_enum(Metrics, CHAT_METRIC_ENUM_NAME, chat_metric)
        return getattr(Metrics, CHAT_METRIC_ENUM_NAME)

    chat_metric_enum = ensure_chat_metric_registered()
    replacement_metric = chat_metric_enum.value

    for task_name, task in pipeline.tasks_dict.items():
        base_name = task_name.split("|", 1)[0]
        if not any(base_name.startswith(prefix) for prefix in prefixes):
            continue
        new_metrics = []
        replaced = False
        for metric in task.metrics:
            if (
                metric.metric_name == "em"
                and metric.category == SamplingMethod.GENERATIVE
            ):
                new_metrics.append(replacement_metric)
                replaced = True
            else:
                new_metrics.append(metric)
        if replaced:
            task.metrics = tuple(new_metrics)
            logger.info("Applied chat metric override to task %s", task_name)
