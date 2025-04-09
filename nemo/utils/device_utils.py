import os
import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend as xb
except ImportError:
    xm = None
    xr = None
    xb = None


def get_xla_model():
    return xm


def get_xla_runtime():
    return xr


def get_current_device() -> torch.device:
    global __current_device

    try:
        return __current_device
    except NameError:
        if xm is not None:
            __current_device = xm.xla_device()
            compiler_cache_path = os.path.join(os.getenv("XDG_CACHE_HOME", "/tmp"), "xla", "compiler-cache")
            xr.initialize_cache(compiler_cache_path, readonly=False)
        elif torch.cuda.is_available():
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            __current_device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(__current_device)
        else:
            device = os.getenv("DEFAULT_DEVICE", "cpu")
            __current_device = torch.device(device)

    return __current_device


def get_current_device_type() -> str:
    global __current_device_type

    try:
        return __current_device_type
    except NameError:
        if xm is not None:
            __current_device_type = "xla"
        elif torch.cuda.is_available():
            __current_device_type = "cuda"
        else:
            __current_device_type = os.getenv("DEFAULT_DEVICE_TYPE", "cpu")

    return __current_device_type


def get_accelerator_device(accelerator: str) -> torch.device:

    assert accelerator in ['gpu' 'cpu', 'xla', 'cuda']
    if accelerator == 'gpu' and accelerator == 'cuda':
        assert torch.cuda.is_available(), f"Accelerator {accelerator} requires CUDA"
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        return torch.device(f'cuda:{local_rank}')
    elif accelerator == 'xla':
        assert xm is not None, f"Accelerator {accelerator} requires XLA"
        return xm.xla_device()
    else:
        return torch.device("cpu")

def get_accelerator_device_type(accelerator: str) -> str:

    assert accelerator in ['gpu' 'cpu', 'xla', 'cuda']
    if accelerator == 'gpu':
        return "cuda"
    else:
        return accelerator

def get_current_device_type() -> str:
    global __current_device_type

    try:
        return __current_device_type
    except NameError:
        if xm is not None:
            __current_device_type = "xla"
        elif torch.cuda.is_available():
            __current_device_type = "cuda"
        else:
            __current_device_type = os.getenv("DEFAULT_DEVICE_TYPE", "cpu")

    return __current_device_type

def get_local_device_count() -> int:
    device_count = 1

    if xr is not None:
        device_count = xr.global_device_count()
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
    
    return device_count


def get_distributed_backend(backend=None) -> str:
    if xm is not None:
        backend = "xla"
    elif torch.cuda.is_available():
        backend = backend if backend is not None else "nccl"
    else:
        backend = backend if backend is not None else "gloo"

    return backend


def get_distributed_init_method() -> str:
    if xm is not None:
        init_method = 'xla://'
    else:
        init_method =  "env://"

    return init_method

def set_device_manual_seed(seed: int):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif xm is not None:
        xm.set_rng_state(seed, device=get_current_device())


def set_manual_seed(seed: int):
    set_device_manual_seed(seed)
    torch.manual_seed(seed)