"""Minimal torch.distributed setup for mpiexec launches on Polaris (PALS/PMI env)."""
import os

import torch
import torch.distributed as dist


def init():
    if "TORCHELASTIC_RUN_ID" in os.environ or ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        # torchrun: its RANK/WORLD_SIZE win over any PMI_* inherited from an outer mpiexec
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = int(os.environ.get("PMI_RANK", 0))
        world = int(os.environ.get("PMI_SIZE", 1))
        local = int(os.environ.get("PMI_LOCAL_RANK", 0))
    if world > 1:
        if "MASTER_ADDR" not in os.environ:
            nodefile = os.environ.get("PBS_NODEFILE")
            os.environ["MASTER_ADDR"] = open(nodefile).readline().strip() if nodefile else "localhost"
        os.environ.setdefault("MASTER_PORT", "29511")
        torch.cuda.set_device(local % torch.cuda.device_count())
        dist.init_process_group("nccl", rank=rank, world_size=world,
                                device_id=torch.device("cuda", local % torch.cuda.device_count()))
    elif torch.cuda.is_available():
        torch.cuda.set_device(local % torch.cuda.device_count())
    return rank, world, local


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def all_reduce_mean(t):
    if dist.is_initialized():
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return t


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_reduce_keyed_mean(sums, counts, dev):
    """Mean of per-key sums over all ranks; keys may differ between ranks (a key absent on a rank adds 0 / 0)."""
    keys = set(sums)
    if dist.is_initialized():
        allk = [None] * dist.get_world_size()
        dist.all_gather_object(allk, sorted(keys))
        keys = set().union(*map(set, allk))
    keys = sorted(keys)
    t = torch.tensor([[sums.get(k, 0.0), counts.get(k, 0)] for k in keys], dtype=torch.float64, device=dev)
    if dist.is_initialized() and len(keys):
        dist.all_reduce(t)
    return {k: float(t[i, 0] / t[i, 1]) for i, k in enumerate(keys) if t[i, 1] > 0}
