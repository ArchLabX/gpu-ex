#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --output=logs/gpu-test-%j.out
#SBATCH --error=logs/gpu-test-%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --partition=rtx-devel

# Replace this with your Unity allocation account.
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

echo "===== Job information ====="
echo "Job ID:             ${SLURM_JOB_ID}"
echo "Node:               ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo

echo "===== NVIDIA-SMI ====="
nvidia-smi
echo

# Load the Python/PyTorch environment available on Unity.
#
# First inspect available modules with:
#   module avail
#
# You may need to adjust this line for the module name provided by Unity.
module load pytorch

echo
echo "===== Python environment ====="
which python
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"

echo
echo "===== Running GPU test ====="
python gpu_test.py --size 4096 --iterations 20