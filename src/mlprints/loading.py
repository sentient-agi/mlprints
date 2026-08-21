"""Model and tokenizer loading with explicit resource lifetimes."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlprints.common.distributed import is_distributed, is_rank0


__all__ = [
    "ModelSession",
    "load_hf_model",
    "load_model",
    "load_model_and_tokenizer",
    "load_tokenizer",
    "open_model",
]


@dataclass
class ModelSession:
    """Own a loaded model and tokenizer for a bounded period."""

    model: Any | None
    tokenizer: Any | None
    role: str = ""

    @property
    def closed(self) -> bool:
        """Whether this session has released its model and tokenizer references."""
        return self.model is None and self.tokenizer is None

    def close(self) -> None:
        """Release owned references and return unused CUDA memory to the allocator."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> ModelSession:
        if self.closed:
            raise RuntimeError("cannot enter a closed model session")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


_BOOL_RUNTIME_OPTIONS = (
    ("use_kernels", False),
    ("compile_model", False),
    ("compile_fullgraph", False),
    ("static_kvcache_for_generation", False),
)


def _model_runtime_options(model_config: dict[str, Any]) -> dict[str, Any]:
    """Parse optional generation-speed knobs from a model YAML mapping."""
    options: dict[str, Any] = {}
    for key, default in _BOOL_RUNTIME_OPTIONS:
        value = model_config.get(key, default)
        if not isinstance(value, bool):
            raise TypeError(f"model {key} must be a bool")
        options[key] = value

    compile_mode = model_config.get("compile_mode", "default")
    if compile_mode is not None and not isinstance(compile_mode, str):
        raise TypeError("model compile_mode must be a string or null")
    compile_dynamic = model_config.get("compile_dynamic")
    if compile_dynamic is not None and not isinstance(compile_dynamic, bool):
        raise TypeError("model compile_dynamic must be a bool or null")
    compile_backend = model_config.get("compile_backend")
    if compile_backend is not None and not isinstance(compile_backend, str):
        raise TypeError("model compile_backend must be a string or null")
    max_cache_len = model_config.get("max_cache_len")
    if max_cache_len is not None and (
        isinstance(max_cache_len, bool)
        or not isinstance(max_cache_len, int)
        or max_cache_len <= 0
    ):
        raise ValueError("model max_cache_len must be a positive int")

    options.update(
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        compile_backend=compile_backend,
        max_cache_len=max_cache_len,
    )
    return options


def _prepare_loaded_model(
    model: Any,
    *,
    is_train: bool,
    use_kernels: bool,
    compile_model: bool,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool | None,
    compile_backend: str | None,
    static_kvcache_for_generation: bool,
    max_cache_len: int | None = None,
) -> Any:
    if use_kernels:
        kernel_model = model
        set_use_kernels = getattr(kernel_model, "set_use_kernels", None)
        if not callable(set_use_kernels):
            kernel_model = getattr(model, "model", None)
            set_use_kernels = getattr(kernel_model, "set_use_kernels", None)
        if not callable(set_use_kernels):
            raise AttributeError(
                "use_kernels=True requires the loaded model (or its wrapped "
                "base model) to implement set_use_kernels()"
            )
        set_use_kernels(True)

    if is_train:
        model.train()
    else:
        model.eval()

    if compile_model:
        compile_kwargs = {
            key: value
            for key, value in {
                "mode": compile_mode,
                "dynamic": compile_dynamic,
                "backend": compile_backend,
                "fullgraph": compile_fullgraph,
            }.items()
            if value is not None
        }
        model = torch.compile(model, **compile_kwargs)

    if static_kvcache_for_generation:
        model.generation_config.cache_implementation = "static"
    if max_cache_len is not None:
        model.generation_config.max_cache_len = max_cache_len

    return model


def load_hf_model(
    path_or_model_id: str,
    *,
    device_map: int | str | torch.device | dict[str, int | str | torch.device] | None = None,
    dtype: str | torch.dtype | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    is_train: bool = False,
    use_kernels: bool = False,
    compile_model: bool = False,
    compile_mode: str | None = "default",
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = None,
    compile_backend: str | None = None,
    static_kvcache_for_generation: bool = False,
    max_cache_len: int | None = None,
) -> Any:
    """Load a Hugging Face model ID or compatible local checkpoint."""
    device_map = "auto" if device_map is None else device_map
    dtype = "bfloat16" if dtype is None else dtype
    model = AutoModelForCausalLM.from_pretrained(
        path_or_model_id,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    return _prepare_loaded_model(
        model,
        is_train=is_train,
        use_kernels=use_kernels,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_backend=compile_backend,
        static_kvcache_for_generation=static_kvcache_for_generation,
        max_cache_len=max_cache_len,
    )


def load_model(
    path_or_model_id: str,
    *,
    device_map: int | str | torch.device | dict[str, int | str | torch.device] | None = None,
    dtype: str | torch.dtype | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    is_train: bool = False,
    use_kernels: bool = False,
    compile_model: bool = False,
    compile_mode: str | None = "default",
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = None,
    compile_backend: str | None = None,
    static_kvcache_for_generation: bool = False,
    max_cache_len: int | None = None,
) -> Any:
    """Load a Hugging Face model/checkpoint or an MLprints attack directory."""
    from mlprints.common.utils import (
        load_implementation,
        load_yaml,
        normalize_str_to_path,
    )

    path = normalize_str_to_path(path_or_model_id)
    attack_config_path = path / "attack.yaml"

    if not attack_config_path.exists():
        return load_hf_model(
            path_or_model_id,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            is_train=is_train,
            use_kernels=use_kernels,
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            compile_backend=compile_backend,
            static_kvcache_for_generation=static_kvcache_for_generation,
            max_cache_len=max_cache_len,
        )

    attack_config = load_yaml(attack_config_path)

    base_config = attack_config.setdefault("base_model_config", {})
    if device_map is not None:
        base_config["device_map"] = device_map
    if dtype is not None and dtype != "auto":
        base_config["dtype"] = dtype
    if attn_implementation is not None:
        base_config["attn_implementation"] = attn_implementation
    if trust_remote_code:
        base_config["trust_remote_code"] = trust_remote_code

    attack_type = attack_config.get("algo", {}).get("name")
    if not attack_type:
        raise ValueError("could not determine attack type from attack directory")

    implementation_path = path / "implementation.py"
    if implementation_path.is_file():
        module = load_implementation(implementation_path)
        attack_class = next(
            value
            for value in vars(module).values()
            if isinstance(value, type) and value.__module__ == module.__name__
        )
    else:
        # Keep this lazy to avoid a loading <-> attack registry import cycle.
        from mlprints.common.attacks import ATTACK_ALGOS

        attack_class = ATTACK_ALGOS[attack_type]["class"]
    model = attack_class.from_config(attack_config)

    return _prepare_loaded_model(
        model,
        is_train=is_train,
        use_kernels=use_kernels,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_backend=compile_backend,
        static_kvcache_for_generation=static_kvcache_for_generation,
        max_cache_len=max_cache_len,
    )


def load_tokenizer(
    path_or_model_id: str,
    *,
    trust_remote_code: bool = False,
) -> Any:
    """Load and validate a local or Hugging Face Hub tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        path_or_model_id,
        trust_remote_code=trust_remote_code,
    )

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer must implement .apply_chat_template()")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must have eos_token_id set")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model_and_tokenizer(
    model_config: dict[str, Any],
    role: str = "",
    is_train: bool = False,
) -> dict[str, Any]:
    """Load a configured model and tokenizer."""
    from mlprints.common.utils import (
        get_context_length_from_model,
        get_context_length_from_tokenizer,
        normalize_str_to_path,
    )

    path_or_model_id = model_config.get("model_id")
    if not path_or_model_id:
        raise ValueError(f"model_id is required for role {role!r}")

    path_or_tokenizer_id = model_config.get("tokenizer_id", path_or_model_id)
    device_map = model_config.get("device_map")
    if is_distributed() and device_map is not None:
        local_device = (
            int(torch.cuda.current_device())
            if torch.cuda.is_available()
            else 0
        )
        if isinstance(device_map, str) and device_map.startswith("cuda:"):
            device_map = f"cuda:{local_device}"
        elif isinstance(device_map, dict):
            device_map = {
                key: local_device if isinstance(value, int) and value >= 0 else value
                for key, value in device_map.items()
            }
    dtype = model_config.get("dtype")
    trust_remote_code = model_config.get("trust_remote_code", False)
    attn_implementation = model_config.get("attn_implementation")
    runtime_options = _model_runtime_options(model_config)

    if is_rank0():
        print(f"Loading {role} model: {path_or_model_id}")

    model = None
    tokenizer = None
    try:
        model_path = normalize_str_to_path(path_or_model_id)
        is_attacked = (model_path / "attack.yaml").exists()
        model = load_model(
            path_or_model_id=path_or_model_id,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            is_train=is_train,
            **runtime_options,
        )
        if is_attacked:
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                raise AttributeError("Attacked model must have tokenizer attribute")
        else:
            tokenizer = load_tokenizer(
                path_or_tokenizer_id,
                trust_remote_code=trust_remote_code,
            )

        if is_train:
            try:
                get_context_length_from_tokenizer(tokenizer)
            except ValueError:
                tokenizer.model_max_length = get_context_length_from_model(model)
    except BaseException:
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    if is_rank0():
        print(f"Loaded {role} model: {path_or_model_id} on {device_map}")

    return {"model": model, "tokenizer": tokenizer}


def open_model(
    model_config: dict[str, Any],
    role: str = "",
    is_train: bool = False,
) -> ModelSession:
    """Load model resources into a closeable context-manager session."""
    loaded = load_model_and_tokenizer(
        model_config,
        role=role,
        is_train=is_train,
    )
    return ModelSession(
        model=loaded["model"],
        tokenizer=loaded["tokenizer"],
        role=role,
    )
