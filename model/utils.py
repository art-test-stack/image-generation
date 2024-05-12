from model.settings import *

import psutil, platform
import torch

def get_gpu():
    if MPS_AVAILABLE:
        torch.mps.empty_cache()
        torch.mps.set_per_process_memory_fraction(0.)
        memory = torch.mps.current_allocated_memory() / 1024 ** 3
        driver_memory = torch.mps.driver_allocated_memory() / 1024 ** 3
        print(f'Using GPU MPS: ({memory:.2f} GB available) driver memory: {driver_memory:.2f} GB')

    elif CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        nb_gpu = torch.cuda.device_count()
        memory = torch.cuda.mem_get_info()[0] / 1024 ** 3
        print(f'{nb_gpu} GPU {"are" if nb_gpu > 1 else "is"} available! Using GPU: "{torch.cuda.get_device_name()}" ({memory:.2f} GB available)')

    else:
        memory = psutil.virtual_memory().available / 1024 ** 3
        print(f'No GPU available... Using CPU: "{platform.processor()}" ({memory:.2f} GB available)')
    