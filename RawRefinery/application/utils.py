import torch

def can_use_gpu():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False