import torch
import deepspeed
from transformers import TrainerCallback

from mlprints.common.distributed import is_rank0, is_zero_param


class EarlyStoppingByLoss(TrainerCallback):
    """
    Stop training once the logged training loss goes below a threshold.
    """

    def __init__(self, loss_threshold: float):
        self.loss_threshold = loss_threshold
        self._saw_loss_log = False

    def on_train_begin(self, args, state, control, **kwargs):
        logging_strategy = args.logging_strategy.value
        if logging_strategy == "no":
            raise ValueError(
                "EarlyStoppingByLoss requires logged training loss. "
                "Set logging_strategy to 'steps' or 'epoch', or remove the callback."
            )
        self._saw_loss_log = False
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_loss = None if logs is None else logs.get("loss")
        if current_loss is None:
            return control

        self._saw_loss_log = True
        if current_loss < self.loss_threshold:
            print(
                f"\n[EarlyStoppingByLoss] Logged training loss {current_loss:.6f} "
                f"is below threshold {self.loss_threshold:.6f}. "
                f"Stopping at step {state.global_step}."
            )
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self._saw_loss_log:
            raise ValueError(
                "EarlyStoppingByLoss requires logged training loss, "
                "but no log with key 'loss' was emitted. "
                "Increase logging frequency or remove the callback."
            )
        return control


class ModelAverageCallback(TrainerCallback):
    """
    Average trainable params toward their initial values at epoch end.
    """
    def __init__(self, model, orig_model_weight: float):
        super().__init__()
        self.orig_weight = orig_model_weight
        # store modifiable parameters in CPU (same precision as original)
        self.orig_state = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if is_zero_param(param):
                    with deepspeed.zero.GatheredParameters(param, modifier_rank=None):
                        self.orig_state[name] = param.detach().cpu().clone()
                else:
                    self.orig_state[name] = param.detach().cpu().clone()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.orig_weight == 0:
            return control

        current_params = dict(kwargs["model"].named_parameters())
        with torch.no_grad():
            for name, orig_param in self.orig_state.items():
                param = current_params[name]
                # in place: param = (1 - orig_weight) * param + orig_weight * orig_param
                if is_zero_param(param):
                    with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                        if is_rank0():
                            param.lerp_(orig_param.to(param.device, param.dtype), self.orig_weight)
                else:
                    param.lerp_(orig_param.to(param.device, param.dtype), self.orig_weight)

        return control
