import os
os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"
import torch

print("PyTorch CUDA version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))