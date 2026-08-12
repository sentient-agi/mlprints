from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import Trainer

from mlprints.common.constants import MASK_LOSS_ID
from mlprints.common.utils import compute_causal_lm_cross_entropy_loss, get_model_device


@dataclass(frozen=True)
class OfflineDistillationLoss:
    """One loss term using precomputed sparse teacher probabilities."""

    mode: str
    weight: float
    temperature: float = 1.0
    scale_by_t_squared: bool = False


@dataclass(frozen=True)
class OnlineDistillationLoss:
    """One full-vocabulary loss term produced by a live frozen teacher."""

    mode: str
    dataset_names: Sequence[str]
    weight: float
    teacher_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None


class CompositeCausalLMTrainer(Trainer):
    """
    Causal LM trainer with hard-label SFT and optional distillation losses.

    Total loss is weighted SFT plus configured offline or online distillation.

    If provided, `dataset_weights` reweight *each* sequence inside train_dataset
    to compute `sft_loss` by its `dataset_name` (required in train_dataset).

    Offline teacher losses read precomputed top-k targets with fields
    (`topk_indices`, `topk_log_probs`, `topk_mask`) from each batch.
    A frozen `online_teacher_model` can instead provide configurable online
    full-vocabulary loss terms.

    NOTE: These targets can be produced by `mlprints.inference.run_inference_logprobs`
    by setting `extract_top_k`.
    """

    OFFLINE_LOSS_MODES = {
        "cross_entropy",
        "positive_deviation",
    }
    ONLINE_LOSS_MODES = {
        "kl",
        "positive_deviation",
    }

    _MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "position_ids"}
    _TOPK_KEYS = {"topk_indices", "topk_log_probs", "topk_mask"}

    def __init__(
        self,
        *args,
        sft_weight: float = 1.0,
        dataset_weights: dict[str, float] | None = None,
        offline_distillation_losses: list[dict | OfflineDistillationLoss] | None = None,
        online_teacher_model=None,
        online_distillation_losses: list[dict | OnlineDistillationLoss] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sft_weight = sft_weight
        self.dataset_weights = dataset_weights
        self.offline_distillation_losses = (
            [
                term
                if isinstance(term, OfflineDistillationLoss)
                else OfflineDistillationLoss(**term)
                for term in offline_distillation_losses
            ]
            if offline_distillation_losses is not None
            else None
        )
        self.online_distillation_losses = (
            [
                term
                if isinstance(term, OnlineDistillationLoss)
                else OnlineDistillationLoss(**term)
                for term in online_distillation_losses
            ]
            if online_distillation_losses is not None
            else None
        )

        if dataset_weights is not None and "dataset_name" not in self.train_dataset.column_names:
            raise ValueError(
                "dataset_weights requires a 'dataset_name' column in train_dataset. "
                "NOTE: Use mlprints.training.concatenate_datasets when building the dataset."
            )

        self.online_teacher_model = online_teacher_model
        if (
            self.sft_weight == 0
            and not self.offline_distillation_losses
            and not self.online_distillation_losses
        ):
            raise ValueError("when sft_weight is 0, at least one active teacher loss is required.")

        if self.offline_distillation_losses is not None:
            if missing_cols := self._TOPK_KEYS - set(self.train_dataset.column_names):
                raise ValueError(
                    "offline_distillation_losses require dataset columns: "
                    f"{sorted(missing_cols)}"
                )
            for term in self.offline_distillation_losses:
                if term.mode not in self.OFFLINE_LOSS_MODES:
                    raise ValueError(f"unsupported offline loss mode: {term.mode}")

        if online_teacher_model is not None:
            if self.offline_distillation_losses is not None or dataset_weights is not None:
                raise ValueError(
                    "online distillation cannot be combined with offline "
                    "distillation or dataset_weights"
                )
            if not self.online_distillation_losses:
                raise ValueError("online distillation requires loss terms")
            self.online_teacher_model.requires_grad_(False)
            self.online_teacher_model.eval()
            for term in self.online_distillation_losses:
                if term.mode not in self.ONLINE_LOSS_MODES:
                    raise ValueError(f"unsupported online loss mode: {term.mode}")
        elif self.online_distillation_losses:
            raise ValueError("online distillation losses require online_teacher_model")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.online_teacher_model is not None:
            return self._online_distillation_loss(
                model,
                inputs,
                return_outputs=return_outputs,
            )

        input_keys = self._MODEL_INPUT_KEYS
        if self.sft_weight != 0 and self.dataset_weights is None:
            input_keys = input_keys | {"labels"} # need labels for SFT loss
        outputs = model(**{
            k: v for k, v in inputs.items()
            if k in input_keys and torch.is_tensor(v)
        })

        loss = torch.zeros((), device=get_model_device(model))

        if self.sft_weight != 0:
            loss += self.sft_weight * self._sft_loss(inputs, outputs)

        if self.offline_distillation_losses is not None:
            for term in self.offline_distillation_losses:
                loss += term.weight * self._offline_distillation_loss(
                    inputs,
                    outputs.logits,
                    term,
                )

        return (loss, outputs) if return_outputs else loss

    def _sft_loss(self, inputs, outputs):
        if self.dataset_weights is None:
            return outputs.loss

        per_sequence_loss = compute_causal_lm_cross_entropy_loss(
            outputs.logits,
            inputs["labels"],
        )
        weights = torch.tensor(
            [self.dataset_weights[name] for name in inputs["dataset_name"]],
            device=per_sequence_loss.device,
            dtype=per_sequence_loss.dtype,
        )
        return (per_sequence_loss * weights).mean()

    def _offline_distillation_loss(self, inputs, logits, term):
        if not inputs["topk_mask"].any():
            return torch.zeros((), device=logits.device)

        scaled_logits = logits / term.temperature

        student_log_probs = torch.gather(
            scaled_logits - torch.logsumexp(scaled_logits, dim=-1, keepdim=True),
            dim=-1,
            index=inputs["topk_indices"],
        )
        teacher_probs = inputs["topk_log_probs"].exp()

        if term.mode == "cross_entropy":
            per_position = -(teacher_probs * student_log_probs).sum(dim=-1)
        elif term.mode == "positive_deviation":
            per_position = F.relu(student_log_probs.exp() - teacher_probs).sum(dim=-1)

        if term.scale_by_t_squared:
            per_position = per_position * (term.temperature ** 2)

        return per_position[inputs["topk_mask"]].mean()

    def _online_distillation_loss(self, model, inputs, *, return_outputs):
        dataset_names = inputs.get("dataset_name")
        if dataset_names is None:
            raise ValueError("online teacher distillation requires dataset_name")

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in self._MODEL_INPUT_KEYS and torch.is_tensor(value)
        }
        outputs = model(**model_inputs)
        student_logits = outputs.logits[:, :-1, :]
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"][:, 1:].bool()
        labels = inputs["labels"]
        assistant_mask = labels[:, 1:].ne(MASK_LOSS_ID)

        teacher_device = get_model_device(self.online_teacher_model)
        teacher_inputs = {
            key: value.to(teacher_device)
            for key, value in model_inputs.items()
        }
        with torch.no_grad():
            teacher_logits = self.online_teacher_model(
                **teacher_inputs
            ).logits[:, :-1, :]
        teacher_logits = teacher_logits.to(student_logits.device)

        configured_names = {
            name
            for term in self.online_distillation_losses
            for name in term.dataset_names
        }
        if unknown := sorted(set(dataset_names) - configured_names):
            raise ValueError(f"no distillation loss configured for datasets: {unknown}")

        loss = student_logits.sum() * 0.0
        for term in self.online_distillation_losses:
            rows = torch.tensor(
                [name in term.dataset_names for name in dataset_names],
                dtype=torch.bool,
                device=student_logits.device,
            )
            mask = attention_mask & assistant_mask & rows.unsqueeze(1)
            if not mask.any() or not term.weight:
                continue

            reference_logits = teacher_logits
            if term.teacher_transform is not None:
                reference_logits = term.teacher_transform(
                    input_ids.to(student_logits.device),
                    teacher_logits.clone(),
                )

            if term.mode == "kl":
                per_position = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.log_softmax(reference_logits, dim=-1),
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)
            elif term.mode == "positive_deviation":
                per_position = F.relu(
                    F.softmax(student_logits, dim=-1)
                    - F.softmax(reference_logits, dim=-1)
                ).sum(dim=-1)

            loss += term.weight * per_position[mask].mean()

        return (loss, outputs) if return_outputs else loss


class CausalLMSFTTrainer(CompositeCausalLMTrainer):
    """Causal LM SFT trainer with optional dataset-weighted hard-label CE."""

    def __init__(
        self,
        *args,
        sft_weight: float = 1.0,
        dataset_weights: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            sft_weight=sft_weight,
            dataset_weights=dataset_weights,
            offline_distillation_losses=None,
            **kwargs,
        )
