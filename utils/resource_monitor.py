import os
import psutil
import torch
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

def get_gpu_usage_mb(device=0):
    if NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return meminfo.used / 1024 / 1024
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    else:
        return 0.0

def get_resource_usage():
    ram = get_ram_usage_mb()
    gpu = get_gpu_usage_mb()
    return ram, gpu