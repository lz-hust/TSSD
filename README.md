# Usage Guide

This guide provides step-by-step instructions to set up the `tssd` project in a Conda environment on a Linux system with CUDA support.

## Prepare TSSD Dataset
[The WAV for TSSD Dataset](https://drive.google.com/drive/folders/15Ay4uwbGOZqfXN9KbBXXlWkXTc4AC-0Y?dmr=1&ec=wgc-drive-hero-goto)

## Installation Steps

### 1. Update Conda
Update Conda to ensure you have the latest version:

```bash
conda update -n base -c defaults conda
```

### 2. Create and Activate Conda Environment
Create a new Conda environment named `mamba` with Python 3.12:

```bash
conda create -n mamba python=3.12
conda activate mamba
```

### 3. Set Environment Variables
Configure environment variables for CUDA 11.8 by adding the following lines to `~/.bashrc`:

```bash
export CUDA_HOME="/usr/local/cuda-11.8/"
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

Apply the changes:

```bash
source ~/.bashrc
```

### 4. Install PyTorch with CUDA Support
Install CUDA Toolkit 11.8 and PyTorch with its dependencies:

```bash
conda install cudatoolkit=11.8 -c nvidia
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

### 5. Install Mamba-SSM 

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba/
```

```bash
pip install .
```

### 6. Install Additional Dependencies
Install additional Python packages required for the project:
```bash
pip install -r requirements.txt
```bash

