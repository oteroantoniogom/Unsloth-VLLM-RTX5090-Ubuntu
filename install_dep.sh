#!/bin/bash
set -euo pipefail

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PS1=${PS1:-}

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   Server Libraries Installation Script   ${NC}"
echo -e "${BLUE}==============================================${NC}"

read -r -p "Do you want to continue with the installation? (y/n): " response
if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${RED}Installation aborted.${NC}"
    exit 0
fi

check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Error occurred${NC}"
        if [ "${1:-}" == "critical" ]; then
            echo -e "${RED}Critical error. Installation cannot continue.${NC}"
            exit 1
        fi
    fi
}

# Only allow root/sudo to install system dependencies
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (sudo).${NC}"
   exit 1
fi

echo -e "\n[STEP] ${YELLOW}Updating repositories and installing basic dependencies...${NC}"
apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    cmake \
    ninja-build \
    gcc-14 g++-14 \
    python3.10 python3.10-venv python3.10-dev \
    unixodbc
check_status "critical"

# Configure GCC 14 and G++ 14 as defaults
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14
check_status

# Detect the real user who executed sudo
REAL_USER=$(logname)

# Define the virtual environment directory in the same folder as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "Environment variables configured successfully."

# Create the virtual environment if it does not exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "\n[STEP] ${YELLOW}Creating virtual environment in $VENV_DIR...${NC}"
    sudo -u "$REAL_USER" python3.10 -m venv "$VENV_DIR"
    check_status "critical"
fi

# Activate the virtual environment and update pip
echo -e "\n[STEP] ${YELLOW}Installing Python packages in the virtual environment...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    pip install --upgrade pip setuptools wheel
"
check_status

# Install individual packages to ensure they do not block each other
echo -e "\n[STEP] ${YELLOW}Installing AI-related packages...${NC}"

for package in unsloth scikit-learn matplotlib ninja cmake wheel pybind11; do
    echo -e "\n[STEP] ${YELLOW}Installing $package...${NC}"
    sudo -u "$REAL_USER" bash -c "
        source $VENV_DIR/bin/activate && \
        pip show $package >/dev/null 2>&1 || pip install $package
    "
    check_status
done

echo -e "\n[STEP] ${YELLOW}Installing PyTorch...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    pip install --force-reinstall torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu128
"
check_status

echo -e "\n[STEP] ${YELLOW}Installing Xformers...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate

    # Remove old installation if it exists
    if [ -d '$SCRIPT_DIR/xformers' ]; then
        echo 'Removing old Xformers installation...'
        rm -rf '$SCRIPT_DIR/xformers'
    fi

    # Clone the Xformers repo from an alternate PR
    echo 'Cloning Xformers from an alternate PR...'
    git clone https://github.com/maludwig/xformers.git '$SCRIPT_DIR/xformers' || { echo '${RED}Error cloning Xformers${NC}'; exit 1; }

    cd '$SCRIPT_DIR/xformers'

    # Update necessary submodules
    git submodule update --init --recursive || { echo '${RED}Error updating Xformers submodules${NC}'; exit 1; }

    # Install required dependencies
    echo 'Installing Xformers dependencies...'
    pip install -r requirements.txt || { echo '${RED}Error installing Xformers dependencies${NC}'; exit 1; }

    # Compile and install Xformers
    echo 'Building and installing Xformers...'
    pip install -v . || { echo '${RED}Error installing Xformers from source${NC}'; exit 1; }

    cd ..
"
check_status

echo -e "\n[STEP] ${YELLOW}Installing vllm...${NC}"
sudo -u "$REAL_USER" bash -c "
    export VLLM_INSTALL_PUNICA_KERNELS=1
    export TORCH_CUDA_ARCH_LIST='12.0'
    export CUDA_HOME=/usr/local/cuda
    export PATH=\$CUDA_HOME/bin:\$PATH
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
    source $VENV_DIR/bin/activate

    # Remove old vllm installation if it exists
    if [ -d '$SCRIPT_DIR/vllm' ]; then
        echo 'Removing old vllm installation...'
        rm -rf '$SCRIPT_DIR/vllm'
    fi

    echo 'Cloning vllm at the commit by @oteroantoniogom...'
    git clone https://github.com/vllm-project/vllm.git '$SCRIPT_DIR/vllm' || { echo '${RED}Error cloning vllm${NC}'; exit 1; }
    cd '$SCRIPT_DIR/vllm'

    # Checkout a specific commit
    git checkout 5d8e1c9279678b3342d9618167121e758ed00c05 || { echo '${RED}Error checking out commit${NC}'; exit 1; }

    cd '$SCRIPT_DIR/vllm'

    echo 'Detecting existing PyTorch installation...'
    python3.10 use_existing_torch.py || { echo '${RED}Error in use_existing_torch.py${NC}'; exit 1; }

    '$VENV_DIR/bin/pip' install -r requirements/build.txt || { echo '${RED}Error installing vllm build dependencies${NC}'; exit 1; }
    '$VENV_DIR/bin/pip' install -r requirements/common.txt || { echo '${RED}Error installing vllm common dependencies${NC}'; exit 1; }

    echo 'Installing vllm...'
    # Determine the number of available cores and set MAX_JOBS to cores-1 (or 1 if only one core is available)
    CORES=\$(nproc)
    if [ \"\$CORES\" -gt 1 ]; then
        MAX_JOBS=\$((CORES - 1))
    else
        MAX_JOBS=1
    fi

    echo \"Using MAX_JOBS=\${MAX_JOBS}\"

    # Use MAX_JOBS for installing vllm
    MAX_JOBS=\${MAX_JOBS} \"$VENV_DIR/bin/pip\" install -e . --no-build-isolation || { echo \"\${RED}Error installing vllm\${NC}\"; exit 1; }

    cd ..
"
check_status

echo -e "\n[STEP] ${YELLOW}Uninstalling PyTorch-triton...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    pip uninstall pytorch-triton -y
"
check_status

echo -e "\n[STEP] ${YELLOW}Installing Triton from the 'patch-1' branch...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate

    # Remove any old Triton installation if it exists
    if [ -d '$SCRIPT_DIR/triton' ]; then
        echo 'Removing old Triton installation...'
        rm -rf '$SCRIPT_DIR/triton'
    fi

    # Clone the Triton repository from your patch-1 branch
    echo 'Cloning Triton from your GitHub on patch-1 branch...'
    git clone --branch patch-1 https://github.com/oteroantoniogom/triton.git '$SCRIPT_DIR/triton' || { echo '${RED}Error cloning Triton${NC}'; exit 1; }

    cd '$SCRIPT_DIR/triton'

    # Update any necessary submodules
    git submodule update --init --recursive || { echo '${RED}Error updating Triton submodules${NC}'; exit 1; }

    # Install needed dependencies
    echo 'Installing Triton dependencies...'
    pip install ninja cmake wheel pybind11 ipywidgets ipykernel chardet openpyxl wandb || { echo '${RED}Error installing Triton dependencies${NC}'; exit 1; }

    # Install Triton from source
    echo 'Building and installing Triton...'
    pip install -e python || { echo '${RED}Error installing Triton from source${NC}'; exit 1; }

    cd ..
"
check_status

echo -e "\n[STEP] ${YELLOW}Reinstalling PyTorch...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    pip install --force-reinstall torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu128
"
check_status

echo -e "\n[STEP] ${YELLOW}Uninstalling PyTorch-triton again...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    pip uninstall pytorch-triton -y
"
check_status

echo -e "\n${GREEN}Installation process completed successfully!${NC}"

# Verify the installations
echo -e "\n${BLUE}==============================================${NC}"
echo -e "${BLUE}   Verifying installations   ${NC}"
echo -e "${BLUE}==============================================${NC}"

# Verify PyTorch within the virtual environment
echo -e "\n[STEP] ${YELLOW}Verifying PyTorch installation...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    python3.10 -c 'import torch; print(\"PyTorch version:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"CUDA version:\", torch.version.cuda if torch.cuda.is_available() else \"N/A\")'
"
check_status

# Verify Triton within the virtual environment
echo -e "\n[STEP] ${YELLOW}Verifying Triton installation...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    python3.10 -c 'import triton; print(\"Triton is installed\")'
"
check_status

# Verify Xformers within the virtual environment
echo -e "\n[STEP] ${YELLOW}Verifying Xformers installation...${NC}"
sudo -u "$REAL_USER" bash -c "
    source $VENV_DIR/bin/activate && \
    python3.10 -c 'import xformers; print(\"Xformers is installed\")'
"
check_status

echo -e "\n${GREEN}All installations have been verified successfully!${NC}"
echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
echo -e "${GREEN}source $VENV_DIR/bin/activate${NC}"
