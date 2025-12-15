import torch

def can_use_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return False
    arch = torch.cuda.get_arch_list()
    major, minor = torch.cuda.get_device_capability()

    if  f"sm_{major}{minor}" in arch:
        print(f"Found CUDA arch {"sm_{major}{minor}"}. Using Cuda")
        return True
    else:
        print(f"Found CUDA arch {"sm_{major}{minor}"}. Must be in {arch}")
        return False