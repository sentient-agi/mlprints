from typing import Any

import torch
import torch.distributed as dist


def is_zero_param(param: Any) -> bool:
    """
    Return True if a parameter is a DeepSpeed ZeRO-3 partitioned parameter.

    Mirrors DeepSpeed's `is_zero_param` method, i.e.
    assumes that partitioned params carry a `ds_id`).
    """
    return torch.is_tensor(param) and hasattr(param, "ds_id")


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_local_rank() -> int:
    """Get local rank (0-based) in distributed mode, 0 otherwise."""
    if is_distributed():
        return dist.get_rank()
    return 0


def is_rank0() -> bool:
    """Check if current process is rank 0 (always True when not distributed)."""
    return get_local_rank() == 0


def get_distributed_info() -> tuple[bool, int, int]:
    """
    Get distributed mode info: (is_distributed, rank, world_size).
    Returns (False, 0, 1) when not distributed.
    """
    if is_distributed():
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


def barrier_if_distributed() -> None:
    """Call dist.barrier() only if in distributed mode."""
    if is_distributed():
        dist.barrier()


def broadcast_from_rank0(obj: Any) -> Any:
    """
    Broadcast an object from rank 0 to all ranks in distributed mode.
    Returns the object unchanged if not in distributed mode.
    """
    if not is_distributed():
        return obj
    
    _, rank, _ = get_distributed_info()
    if rank == 0:
        container = [obj]
    else:
        container = [None]
    dist.broadcast_object_list(container, src=0)
    return container[0]
