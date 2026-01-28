import torch

def is_oom_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "out of memory" in msg
        or "cuda error" in msg and "memory" in msg
        or isinstance(err, torch.cuda.OutOfMemoryError)
    )
