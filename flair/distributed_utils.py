import logging
import os

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group

import flair

log = logging.getLogger("flair")


def launch_distributed(fp, launch_args=None, on_close=None, *args, **kwargs):
    """Executes the function fp(*args) on multiple GPUs (all local GPUs)."""
    world_size = torch.cuda.device_count()
    world_size = 2#torch.cuda.device_count()
    log.info(f"Launching {world_size} processes")
    parent_conn, child_conn = mp.Pipe()
    mp.spawn(_entrypoint, args=(world_size, launch_args, on_close, child_conn, fp, args, kwargs), nprocs=world_size)
    return_value = parent_conn.recv()
    return return_value


# def entrypoint(rank, world_size, fp, *args):
def _entrypoint(rank, world_size, launch_args, on_end, return_values, fp, args, kwargs):
    log.info(f"Started process on rank={rank}")
    _ddp_setup(rank, world_size)
    return_value = fp(*args, **kwargs)
    if is_main_process():
        return_values.send(return_value)
    # on_end(launch_args)
    destroy_process_group()


def _ddp_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    flair.device = torch.device(rank)
    torch.cuda.set_device(flair.device)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def is_main_process() -> bool:
    """True for exactly 1 process, regardless of whether being run on CPU/single-GPU/multi-gpu."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def aggregate_across_processes(value, f):
    output = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output, value)
    return f(output)


class DistributedModel(torch.nn.parallel.DistributedDataParallel):
    """DistributedDataParallel, but redirects access to methods and attributes to the original Model."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
