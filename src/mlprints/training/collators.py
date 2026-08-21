from dataclasses import dataclass
from typing import Any

import torch

from mlprints.common.constants import MASK_LOSS_ID


@dataclass
class CausalLMPadOnlyDataCollator:
    """
    Collator that pads features to the right to the multiple of `pad_to_multiple_of`.

    If `pad_to_multiple_of` is None, pads sequences to maximum length within a batch.
    """

    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return self._pad_batch(features, tensor_fields={})

    def _pad_batch(
        self,
        features: list[dict[str, Any]],
        tensor_fields: dict[str, torch.dtype],
    ) -> dict[str, torch.Tensor]:
        dataset_names = (
            [feature["dataset_name"] for feature in features]
            if "dataset_name" in features[0]
            else None
        )
        labels_list = (
            [feature["labels"] for feature in features]
            if "labels" in features[0]
            else None
        )
        tensor_values = {
            key: [
                torch.as_tensor(feature[key], dtype=dtype)
                for feature in features
            ]
            for key, dtype in tensor_fields.items()
        }
        model_input_names = set(
            getattr(
                self.tokenizer,
                "model_input_names",
                ("input_ids", "attention_mask"),
            )
            or ("input_ids", "attention_mask")
        )
        model_input_names.update(("input_ids", "attention_mask"))
        model_input_names.difference_update(
            {"labels", "dataset_name", *tensor_fields}
        )
        features_for_pad = [
            {
                key: value
                for key, value in feature.items()
                if key in model_input_names
            }
            for feature in features
        ]

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            batch = self.tokenizer.pad(
                features_for_pad,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        padded_len = batch["input_ids"].shape[1]
        if labels_list is not None:
            batch["labels"] = torch.tensor(
                [
                    labels + [MASK_LOSS_ID] * (padded_len - len(labels))
                    for labels in labels_list
                ],
                dtype=torch.long,
            )
        if dataset_names is not None:
            batch["dataset_name"] = dataset_names

        for key, values in tensor_values.items():
            shape = (len(values), padded_len, *values[0].shape[1:])
            batch[key] = torch.zeros(shape, dtype=values[0].dtype)
            for i, value in enumerate(values):
                batch[key][i, : value.shape[0]] = value

        return batch


@dataclass
class TopKCausalLMDataCollator(CausalLMPadOnlyDataCollator):
    """
    Collator that pads features via CausalLMPadOnlyDataCollator and
    adds (potentially sparse) top-k teacher targets and a mask to each token position.

    The mask is True for positions that have a top-k teacher target.

    Each sample should contain `topk_indices` [L, K], `topk_log_probs` [L, K],
    and `topk_mask` [L].
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return self._pad_batch(
            features,
            tensor_fields={
                "topk_indices": torch.long,
                "topk_log_probs": torch.float,
                "topk_mask": torch.bool,
            },
        )
