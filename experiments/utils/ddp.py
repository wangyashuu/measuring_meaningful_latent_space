import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group


def ddp_dataloader(loader):
    params = dict(pin_memory=True, shuffle=False, drop_last=True)
    shuffle = loader.sampler.__class__ == RandomSampler
    sampler = DistributedSampler(
        loader.dataset, shuffle=shuffle, drop_last=True
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        sampler=sampler,
        **params
    )


def ddp_model(model, device):
    return DDP(model.to(device), device_ids=[device])


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def ddp_destory():
    destroy_process_group()


def run_func(rank, world_size, func, *args):
    ddp_setup(rank, world_size)
    func(rank, *args)
    ddp_destory()


def ddp_run(func, *args):
    world_size = torch.cuda.device_count()
    mp.spawn(run_func, args=(world_size, func) + args, nprocs=world_size)
