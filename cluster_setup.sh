#!/bin/bash
# Oxford HTC Cluster Setup Script for MFS Project
# Run this once after logging into the cluster

echo "=== Oxford HTC Cluster Setup for MFS ==="

# Check if we're on the right cluster
if [[ $(hostname) != *"htc"* ]] && [[ $(hostname) != *"arc"* ]]; then
    echo "Warning: This doesn't appear to be the Oxford HTC cluster"
    echo "Current hostname: $(hostname)"
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p ~/CryptoSafety/logs
mkdir -p ~/CryptoSafety/outputs
mkdir -p ~/CryptoSafety/checkpoints
mkdir -p ~/.cache/huggingface
mkdir -p ~/.wandb

# Set up environment variables
echo "Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# MFS Project Environment Variables
export MFS_PROJECT_ROOT=~/CryptoSafety
export CUDA_CACHE_PATH=~/.cache/cuda
export HF_CACHE_DIR=~/.cache/huggingface
export WANDB_CACHE_DIR=~/.wandb
export PYTHONPATH=$MFS_PROJECT_ROOT:$PYTHONPATH

# Oxford HTC specific optimizations
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Helpful aliases
alias mfs='cd $MFS_PROJECT_ROOT'
alias sbatch-mfs='cd $MFS_PROJECT_ROOT && sbatch'
alias squeue-mfs='squeue -u $USER'
alias scancel-all='scancel -u $USER'
alias gpu-status='sinfo -p gpu,contrib-gpu --format="%.18N %.9P %.6t %.14C %.8m %.8G %.15f"'

EOF

# Source the updated bashrc
source ~/.bashrc

# Check available modules
echo "=== Available modules ==="
module avail python
module avail cuda
module avail gcc

# Load default modules for MFS
echo "=== Loading default modules ==="
module purge
module load python/3.9 2>/dev/null || module load python/3.8 || echo "No suitable Python module found"
module load cuda/11.8 2>/dev/null || module load cuda/11.7 || module load cuda/11.2 || echo "No suitable CUDA module found"
module load gcc/9.3.0 2>/dev/null || module load gcc || echo "No GCC module found"

# Show loaded modules
echo "=== Currently loaded modules ==="
module list

# Check disk quotas
echo "=== Disk quotas ==="
quota -u 2>/dev/null || echo "Quota command not available"
df -h ~ | tail -1
df -h /tmp | tail -1

# Check SLURM configuration
echo "=== SLURM configuration ==="
echo "Available partitions:"
sinfo --format="%.12P %.5a %.10l %.6D %.6t %.8N %.15G"

echo "=== GPU information ==="
sinfo -p gpu,contrib-gpu --format="%.18N %.9P %.6t %.14C %.8m %.8G %.15f" 2>/dev/null || echo "No GPU partition found"

# Create a simple job submission script template
cat > ~/submit_template.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=univ5678@ox.ac.uk

# Your commands here
module purge
module load python/3.9 cuda/11.8

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"

# Your Python script
# python my_script.py

echo "Job completed at $(date)"
EOF

echo "=== Setup complete! ==="
echo "Key files created:"
echo "  - ~/submit_template.slurm (SLURM job template)"
echo "  - Updated ~/.bashrc with MFS environment"
echo ""
echo "To get started:"
echo "  1. source ~/.bashrc  # Reload environment"
echo "  2. cd ~/CryptoSafety  # Go to project directory"
echo "  3. sbatch mfs_training_test.slurm  # Test submission"
echo ""
echo "Useful commands:"
echo "  - squeue-mfs          # Check your jobs"
echo "  - gpu-status          # Check GPU availability"
echo "  - mfs                 # Go to project directory"
echo ""
echo "Next steps:"
echo "  1. Run the test job: sbatch mfs_training_test.slurm"
echo "  2. If test passes, run: sbatch mfs_training.slurm" 