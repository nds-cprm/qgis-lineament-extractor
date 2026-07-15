import torch


def identify_device():
    if getattr(torch, "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
        
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
        
    else:
        return torch.device("cpu")

def get_device_details(device):
    match device.type:
        case "cuda":
            # Check if device is ROCm (AMD) or native CUDA (NVIDIA)
            is_rocm = "rocm" in torch.__version__
            device_type = "AMD (ROCm)" if is_rocm else "NVIDIA (CUDA)"
            gpu_name = torch.cuda.get_device_name(0)

            return f"Detected GPU: {gpu_name}, Plataform: {device_type}"
        
        case "mps":
            return "Detected GPU: Apple Silicon (MPS)"
        
        case _:
            return "No known GPU device found. Using %s." % device.type