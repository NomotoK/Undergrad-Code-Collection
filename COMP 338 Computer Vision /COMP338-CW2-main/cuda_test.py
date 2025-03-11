import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use GPU.")
    print("Current GPU device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. PyTorch will use CPU.")