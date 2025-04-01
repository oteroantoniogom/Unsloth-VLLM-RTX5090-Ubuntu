# 🚀 AI Environment Setup: Unsloth + vLLM (GPU Training)
This repository contains an automated bash script (install_env.sh) designed to easily and correctly set up all necessary dependencies and libraries to run GPU-accelerated training using Unsloth and vLLM on Ubuntu Linux.

The script prepares your environment specifically optimized for CUDA-compatible NVIDIA GPUs, ensuring seamless installation of PyTorch nightly, Triton, Xformers, and other key ML packages.

## 🎯 Main Objective
Ensure a reliable and fully operational installation of all dependencies and libraries required to efficiently run Unsloth integrated with vLLM, leveraging GPU acceleration for state-of-the-art training and inference workloads on Ubuntu.

## ✅ What Does This Script Do?
Updates system dependencies:
Installs necessary build tools and compilers (gcc-14, g++-14, cmake, ninja-build, etc.).

Python 3.10 virtual environment:
Automatically creates or uses an existing virtual environment (venventrenar) in your current directory.

Installs GPU-optimized ML libraries:

PyTorch nightly (CUDA 12.8 support)

Unsloth for efficient training & inference

vLLM from a specific commit optimized for inference

Triton from a custom branch (for optimized kernels)

Xformers (efficient transformer architectures)

Supporting libraries: scikit-learn, matplotlib, pybind11, etc.

Verifies each installation after setup by importing and testing key Python modules.

## 📋 Requirements
OS: Ubuntu or Debian-based Linux

Administrative Privileges: Root access (or sudo) to install system packages

GPU & CUDA Toolkit: NVIDIA GPU with CUDA drivers installed (CUDA 12.8 recommended)

Internet: Needed to clone repositories and install Python packages

## 🚀 Quickstart Guide
Follow these steps to quickly set up your AI training environment:

Clone this repository or download the script:

bash
Copiar
Editar
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Make the script executable:

bash
Copiar
Editar
chmod +x install_env.sh
Run the script with sudo privileges:

bash
Copiar
Editar
sudo ./install_env.sh
Follow on-screen instructions.

Activate the Python virtual environment after installation:

bash
Copiar
Editar
source venventrenar/bin/activate
🔎 Verifying Installation
The script automatically performs verification steps at the end of the installation:

Checks PyTorch version and CUDA availability.

Confirms successful imports of Unsloth, Triton, and Xformers.

To manually verify afterward run, for instance, on a Jupyter Notebook:
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
print(torch.__version__)  
print(torch.cuda.is_available())  # Debe devolver True
print(torch.cuda.device_count())  # Debe mostrar al menos 1
print(torch.cuda.get_device_name(0))  # Debe mostrar el nombre de tu GPU

max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

## Customizing the Setup
CUDA versions:
Modify PyTorch installation URLs or CUDA variables inside the script if your CUDA version differs from the recommended CUDA 12.8.

Repository Forks & Commits:
You can easily point to official repositories or alternative branches/commits by modifying URLs and commit hashes inside the script.

## 📄 License
This script is provided under the MIT License. Feel free to modify and distribute, following license guidelines.

## 📞 Support & Issues
If you encounter any problems or have suggestions, please open an issue in this repository or submit a PR for enhancements.

Happy GPU-accelerated training with Unsloth & vLLM! 🚀✨
