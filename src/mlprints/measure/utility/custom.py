"""Benchmark-specific LightEval customizations."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import logging
import re
import string
from types import ModuleType
from typing import Any, Dict, Iterable, Tuple

from aenum import extend_enum
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import (
    ExactMatches,
    PassAtK,
    SampleLevelComputation,
)
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import Pipeline
from lighteval.tasks.lighteval_task import LightevalTaskConfig
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

_GPQA_PATTERNS = (
    re.compile(
        r"^\s*(?:response\s*:\s*)?"
        r"(?:final\s+answer|answer)\s*[:\-]\s*"
        r"(?:answer\s*[:\-]\s*)?"
        r"([A-D])(?=\s|[\)\]\(\[.,:;\-]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:the\s+correct\s+answer\s+is|option)\s+"
        r"([A-D])(?=\s|[\)\]\(\[.,:;\-]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*([A-D])\s*\)", re.IGNORECASE),
)
_NUMBER = (
    r"[-+]?(?:\$\s*)?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?"
)
_GSM_EXPLICIT_PATTERNS = (
    re.compile(rf"####\s*({_NUMBER})", re.IGNORECASE),
    re.compile(
        rf"^\s*(?:final\s+)?answer\s*[:=\-]\s*"
        rf"({_NUMBER})",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"\\boxed\{\s*(" + _NUMBER + r")\s*\}"),
)
_GSM_CONCLUSION_CUE = re.compile(
    r"\b(?:"
    r"therefore|thus|hence|accordingly|finally|"
    r"in\s+(?:total|conclusion|summary)|"
    r"the\s+(?:final\s+)?(?:answer|result|total)\s+is"
    r")\b",
    re.IGNORECASE,
)
GPQA_DIAMOND_METRIC_NAME = "gpqa_pass@k:k=1"
GSM8K_METRIC_NAME = "extractive_match"


def mean_or_zero(values: Iterable[float]) -> float:
    """Arithmetic mean of ``values`` (empty → 0.0)."""
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def canonical_triviaqa_normalizer(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_prediction(text: str) -> str:
    """Canonical TriviaQA normalization, after stripping explicit answer wrappers."""
    text = re.sub(
        r"^(?:answer|final answer|response)\s*[:\-]\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return canonical_triviaqa_normalizer(text)


def mlprints_triviaqa_prompt(line: dict, task_name: str | None = None) -> Doc:
    """Keep every distinct non-empty canonical TriviaQA alias."""
    aliases = []
    seen: set[str] = set()

    for alias in line["answer"]["aliases"]:
        normalized = canonical_triviaqa_normalizer(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            aliases.append(alias)

    if not aliases:
        raise ValueError("TriviaQA question has no non-empty aliases")

    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        gold_index=0,
        choices=[aliases],
    )


def mlprints_triviaqa_metric() -> SampleLevelMetric:
    """Exact match with symmetric canonical TriviaQA normalization."""
    return SampleLevelMetric(
        metric_name="em",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=ExactMatches(
            normalize_gold=canonical_triviaqa_normalizer,
            normalize_pred=normalize_prediction,
            strip_strings=True,
            type_exact_match="full",
        ),
        corpus_level_fn=mean_or_zero,
    )


def mlprints_triviaqa_config(base: LightevalTaskConfig) -> LightevalTaskConfig:
    """Corrected TriviaQA task: all aliases, canonical EM, newline-only stop."""
    return replace(
        base,
        prompt_function=mlprints_triviaqa_prompt,
        metrics=(mlprints_triviaqa_metric(),),
        stop_sequence=["\n"],
    )


@contextmanager
def _override_named_task(
    module: ModuleType,
    name: str,
    factory: Callable[[LightevalTaskConfig], LightevalTaskConfig],
) -> Iterator[LightevalTaskConfig]:
    original = None
    index = None
    for i, config in enumerate(module.TASKS_TABLE):
        if config.name == name:
            original = config
            index = i
            break
    if original is None or index is None:
        raise LookupError(f"LightEval task {name!r} was not found")

    aliases = [
        attr
        for attr, value in vars(module).items()
        if value is original and not attr.startswith("_")
    ]
    corrected = factory(original)
    module.TASKS_TABLE[index] = corrected
    for attr in aliases:
        setattr(module, attr, corrected)
    try:
        yield corrected
    finally:
        module.TASKS_TABLE[index] = original
        for attr in aliases:
            setattr(module, attr, original)


def _response_text(model_response: ModelResponse) -> str:
    seq = model_response.final_text or getattr(model_response, "text", None) or []
    return seq[0] if seq else ""


def extract_gpqa_answer(text: str) -> str | None:
    """Return the last anchored A–D choice, ignoring unanchored reasoning letters."""
    matches = []

    for line in text.splitlines():
        for pattern in _GPQA_PATTERNS:
            match = pattern.match(line)
            if match:
                matches.append(match.group(1).upper())
                break

    return matches[-1] if matches else None


class GPQAAnchoredExactMatch(SampleLevelComputation):
    """Score GPQA from an explicit anchored letter, never unanchored A–D scans."""

    def __init__(self) -> None:
        self.extract = extract_gpqa_answer

    def compute(
        self,
        doc: Doc,
        model_response: ModelResponse,
        **kwargs,
    ) -> float:
        predicted = self.extract(_response_text(model_response))
        gold = doc.get_golds()[0]
        return float(predicted is not None and predicted == gold)


def mlprints_gpqa_metric(k: int = 1) -> SampleLevelMetric:
    """Anchored GPQA accuracy; wrap in PassAtK only when ``k > 1``."""
    scorer = GPQAAnchoredExactMatch()
    if k > 1:
        sample_level_fn: SampleLevelComputation = PassAtK(
            k=k,
            sample_scoring_function=scorer,
        )
        metric_name = f"gpqa_pass@k:k={k}"
    else:
        sample_level_fn = scorer
        metric_name = GPQA_DIAMOND_METRIC_NAME
    return SampleLevelMetric(
        metric_name=metric_name,
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=sample_level_fn,
        corpus_level_fn=mean_or_zero,
    )


def mlprints_gpqa_config(base: LightevalTaskConfig) -> LightevalTaskConfig:
    """Corrected GPQA Diamond metric: anchored letter match, leaderboard name."""
    k = 1
    for metric in base.metrics:
        sample_k = getattr(metric.sample_level_fn, "k", None)
        if sample_k is not None:
            k = int(sample_k)
            break
    return replace(base, metrics=(mlprints_gpqa_metric(k=k),))


def normalize_number(value: str) -> Decimal | None:
    value = (
        value.replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace(",", "")
        .replace(" ", "")
        .rstrip(string.punctuation)
    )
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def extract_gold_after_marker(text: str) -> Decimal | None:
    """Official GSM8K gold: the number after the final ``####`` marker."""
    if "####" not in text:
        return None
    suffix = text.rsplit("####", 1)[1]
    numbers = re.findall(_NUMBER, suffix)
    if not numbers:
        return None
    return normalize_number(numbers[0])


def extract_gsm8k_answer(text: str) -> Decimal | None:
    """Extract exactly one GSM8K final answer, never every in-reasoning number."""
    explicit = [
        (match.start(), match.group(1))
        for pattern in _GSM_EXPLICIT_PATTERNS
        for match in pattern.finditer(text)
    ]
    if explicit:
        explicit.sort(key=lambda item: item[0])
        return normalize_number(explicit[-1][1])

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    if not paragraphs:
        return None

    final_paragraph = paragraphs[-1]
    final_numbers = re.findall(_NUMBER, final_paragraph)
    if not final_numbers:
        return None
    if len(final_numbers) > 1 and not _GSM_CONCLUSION_CUE.search(final_paragraph):
        return None

    return normalize_number(final_numbers[-1])


class GSM8KFinalAnswerMatch(SampleLevelComputation):
    """Score GSM8K from one final numeric answer against the official marker gold."""

    def __init__(self) -> None:
        self.extract_pred = extract_gsm8k_answer
        self.extract_gold = extract_gold_after_marker

    def compute(
        self,
        doc: Doc,
        model_response: ModelResponse,
        **kwargs,
    ) -> float:
        predicted = self.extract_pred(_response_text(model_response))
        gold = self.extract_gold(doc.get_golds()[0])
        return float(predicted is not None and predicted == gold)


def mlprints_gsm8k_metric() -> SampleLevelMetric:
    """GSM8K extractive match over a single final answer."""
    return SampleLevelMetric(
        metric_name=GSM8K_METRIC_NAME,
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=GSM8KFinalAnswerMatch(),
        corpus_level_fn=mean_or_zero,
    )


def mlprints_gsm8k_config(base: LightevalTaskConfig) -> LightevalTaskConfig:
    """Corrected GSM8K metric: one final answer, not any in-reasoning expression."""
    return replace(base, metrics=(mlprints_gsm8k_metric(),))


@contextmanager
def override_triviaqa_task() -> Iterator[LightevalTaskConfig]:
    """Replace LightEval's TriviaQA config for the duration of pipeline construction."""
    import lighteval.tasks.tasks.triviaqa as triviaqa_module

    with _override_named_task(
        triviaqa_module,
        "triviaqa",
        mlprints_triviaqa_config,
    ) as corrected:
        logger.info(
            "Overrode LightEval TriviaQA with canonical exact match "
            "and newline-only stopping"
        )
        yield corrected


@contextmanager
def override_gpqa_task() -> Iterator[LightevalTaskConfig]:
    """Replace LightEval's GPQA Diamond metric for the duration of pipeline construction."""
    import lighteval.tasks.tasks.gpqa as gpqa_module

    with _override_named_task(
        gpqa_module,
        "gpqa:diamond",
        mlprints_gpqa_config,
    ) as corrected:
        logger.info(
            "Overrode LightEval GPQA Diamond with anchored letter exact match"
        )
        yield corrected


@contextmanager
def override_gsm8k_task() -> Iterator[LightevalTaskConfig]:
    """Replace LightEval's GSM8K metric for the duration of pipeline construction."""
    import lighteval.tasks.tasks.gsm8k as gsm8k_module

    with _override_named_task(
        gsm8k_module,
        "gsm8k",
        mlprints_gsm8k_config,
    ) as corrected:
        logger.info(
            "Overrode LightEval GSM8K with single-final-answer extractive match"
        )
        yield corrected


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
