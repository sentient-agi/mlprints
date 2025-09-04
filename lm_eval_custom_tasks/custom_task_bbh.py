# /your/path/my_tasks/bbh_overrides.py
from lm_eval.api.registry import register_task, TASK_REGISTRY
from lm_eval.tasks import get_task
BaseCandidate = get_task("bbh_cot_fewshot")
BaseTask = BaseCandidate if isinstance(BaseCandidate, type) else BaseCandidate()

@register_task("bbh_cot_fewshot_no_dnl")
class BBHCotFewshotNoDNL(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = getattr(self, "config", None) or getattr(self, "_config", None)
        if not cfg:
            return
        gen_kwargs = getattr(cfg, "generation_kwargs", None)
        if gen_kwargs is None and isinstance(cfg, dict):
            gen_kwargs = cfg.get("generation_kwargs")
        if isinstance(gen_kwargs, dict):
            # Example: remove the double newline stop and bump max tokens
            until = gen_kwargs.get("until")
            if until:
                gen_kwargs["until"] = [s for s in until if s != "\n\n"]
            # Optional: change length limit here
            # gen_kwargs["max_new_tokens"] = 512  # HF
            # gen_kwargs["max_tokens"] = 512      # OpenAI/vLLM