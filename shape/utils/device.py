"""Runtime device selection helpers."""

import os

import torch


def resolve_torch_device():
    """Return a device that can actually allocate a tensor."""
    allocation = (
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"SLURM_JOB_GPUS={os.getenv('SLURM_JOB_GPUS', '<unset>')}"
    )
    if not torch.cuda.is_available():
        return torch.device("cpu"), (
            f"CUDA unavailable in PyTorch "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"{allocation})"
        )

    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda"), None
    except Exception as exc:
        return torch.device("cpu"), (
            f"CUDA tensor allocation failed ({allocation}): {exc}"
        )


def cellpose_precision_kwargs(device):
    """Use BF16 only on GPUs with practical native BF16 support."""
    if device.type != "cuda":
        return {"use_bfloat16": False}

    major, _ = torch.cuda.get_device_capability(device)
    return {"use_bfloat16": major >= 8}
