# 🚀 AI Environment Setup: Unsloth + vLLM (GPU Training)

This repository contains an automated bash script (**`install_dep.sh`**) designed to quickly and reliably set up all dependencies required to run GPU-accelerated training using **Unsloth** and **vLLM** on Ubuntu Linux.

Specifically optimized for CUDA-compatible NVIDIA GPUs, the script ensures seamless installation of PyTorch nightly, Triton, Xformers, and other essential ML packages.

---

## 🎯 Main Objective

Provide a reliable, fully operational environment to efficiently run **Unsloth** integrated with **vLLM**, enabling GPU acceleration for cutting-edge training and inference workloads on Ubuntu.

---

## ✅ What Does This Script Do?

- **Updates System Dependencies**  
  Installs build tools and compilers (`gcc-14`, `g++-14`, `cmake`, `ninja-build`, etc.).

- **Sets Up Python 3.10 Virtual Environment**  
  Creates or reuses a virtual environment (`venv`) in your current directory.

- **Installs GPU-Optimized ML Libraries**:
  - **PyTorch Nightly** (CUDA 12.8)
  - **Unsloth** (efficient training & inference)
  - **vLLM** (fast LLM inference)
  - **Triton** (custom optimized kernels)
  - **Xformers** (efficient transformer operations)
  - Additional ML libraries: `scikit-learn`, `matplotlib`, `pybind11`

- **Verification of Installation**  
  Confirms correct installation by importing key Python modules automatically.

---

## 📋 Requirements

- **OS**: Ubuntu or Debian-based Linux  
- **Privileges**: `sudo` access (to install system packages)  
- **GPU & CUDA**: NVIDIA GPU with CUDA drivers installed (recommended CUDA 12.8)  
- **Internet**: Required for cloning repositories and downloading packages

---

## 🚀 Quickstart Guide

Follow these steps to set up your GPU training environment quickly:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
````

### 2. Make the Script Executable
```bash
chmod +x install_dep.sh
```

### 3. Run the Instalation Script
```bash
sudo ./install_dep.sh
```

### 4. Activate the Virtual Environment
```bash
source venv/bin/activate
```

## 🔎 Verifying Installation
The script automatically verifies installations by checking PyTorch and CUDA status, as well as imports for Unsloth, Triton, and Xformers.

To manually verify afterward (e.g., in a Jupyter Notebook):
```python
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch

print(torch.__version__)  
print(torch.cuda.is_available())        # Should return True
print(torch.cuda.device_count())        # Should show at least 1
print(torch.cuda.get_device_name(0))    # Should show your GPU name

max_seq_length = 1024  # Adjust as needed
lora_rank = 64         # Higher = smarter but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,          # False for LoRA 16-bit
    fast_inference=True,        # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Common values: 8, 16, 32, 64, 128
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Modify modules if needed
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

## ⚙️ Customizing the Setup
CUDA Versions:
Modify installation URLs or CUDA variables inside install_dep.sh if your CUDA version differs from the recommended CUDA 12.8.

Repository Branches & Commits:
You can easily point to official repositories or custom branches/commits by modifying URLs and commit hashes within the script.

## 📄 License
This project is licensed under the MIT License. See LICENSE for details.

## 📞 Support & Issues
Encounter any issues or have suggestions? Feel free to open an issue or submit a PR for improvements.

---

Happy GPU-accelerated training with Unsloth & vLLM! 🚀✨
