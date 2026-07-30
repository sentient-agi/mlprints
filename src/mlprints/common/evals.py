from collections import defaultdict
from typing import Any

from lighteval.tasks.registry import Registry


_TASK_SPECIFIC_KEYS = {
    "n_shots",
    "max_samples",
    "batch_size",
    "generation_size"
}


def resolve_task_params(
    tasks: str,
    *,
    task_overrides: dict[str, dict[str, Any] | None] | None = None,
    default_max_samples: Any = None,
    default_batch_size: Any = None,
    default_generation_size: Any = None,
) -> list[dict[str, Any]]:
    """Resolve LightEval tasks and group those with equal runtime settings.

    Overrides use expanded task names and may set ``n_shots``, ``max_samples``,
    ``batch_size``, or ``generation_size``.
    """
    task_overrides = dict(task_overrides or {})
    resolved: list[tuple[str, str, int | None, int]] = []
    seen: set[str] = set()

    entries = [entry.strip() for entry in tasks.split(",") if entry.strip()]
    if not entries:
        raise ValueError("tasks must contain at least one task entry")

    for entry in entries:
        task_expr, separator, fewshot = entry.partition("|")
        task_expr = task_expr.strip()
        if not task_expr or "|" in fewshot:
            raise ValueError(
                f"Invalid task entry {entry!r}. Expected 'task' or 'task|fewshot'."
            )
        explicit_n_shots = int(fewshot.strip()) if separator else None
        _, marker, metric_params = task_expr.partition("@")
        metric_suffix = f"@{metric_params}" if marker else ""

        loaded = Registry(tasks=task_expr).load_tasks()
        if not loaded:
            raise ValueError(f"Task entry '{task_expr}' did not resolve to any tasks")

        for full_name in loaded:
            base_name, default_n_shots = full_name.rsplit("|", 1)
            if base_name in seen:
                raise ValueError(f"Task '{base_name}' is selected more than once")
            seen.add(base_name)
            resolved.append(
                (base_name, metric_suffix, explicit_n_shots, int(default_n_shots))
            )

    unknown = set(task_overrides) - seen
    if unknown:
        raise ValueError(
            f"task_overrides references tasks not in tasks: {sorted(unknown)}"
        )

    for name, ovr in task_overrides.items():
        if ovr is None:
            continue
        if not isinstance(ovr, dict):
            raise ValueError(f"task_overrides['{name}'] must be a dict or null")
        bad_keys = set(ovr) - _TASK_SPECIFIC_KEYS
        if bad_keys:
            raise ValueError(
                f"task_overrides['{name}'] has unsupported keys: {sorted(bad_keys)}"
            )

    grouped = defaultdict(list)
    for base_name, metric_suffix, explicit_n_shots, default_n_shots in resolved:
        ovr = task_overrides.get(base_name) or {}
        ovr_n_shots = ovr.get("n_shots")
        if ovr_n_shots is not None:
            ovr_n_shots = int(ovr_n_shots)
        if (
            explicit_n_shots is not None
            and ovr_n_shots is not None
            and ovr_n_shots != explicit_n_shots
        ):
            raise ValueError(
                f"Conflicting few-shot for '{base_name}': "
                f"tasks says |{explicit_n_shots}, override says n_shots={ovr_n_shots}."
            )

        n_shots = (
            ovr_n_shots
            if ovr_n_shots is not None
            else explicit_n_shots
            if explicit_n_shots is not None
            else default_n_shots
        )
        detail = {
            "base_name": base_name,
            "task_entry": f"{base_name}{metric_suffix}|{n_shots}",
            "n_shots": n_shots,
            "max_samples": ovr.get("max_samples", default_max_samples),
            "batch_size": ovr.get("batch_size", default_batch_size),
            "generation_size": ovr.get("generation_size", default_generation_size),
        }
        grouped[
            (detail["max_samples"], detail["batch_size"], detail["generation_size"])
        ].append(detail)

    return [
        {
            "tasks": ",".join(t["task_entry"] for t in group),
            "task_details": group,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "generation_size": generation_size,
        }
        for (max_samples, batch_size, generation_size), group in grouped.items()
    ]
