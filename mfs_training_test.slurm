#!/bin/bash
#SBATCH --job-name=mfs-test
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --output=logs/mfs_test_%j.out
#SBATCH --error=logs/mfs_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=univ5678@ox.ac.uk

# Test script for validating MFS setup on Oxford HTC cluster

echo "=== MFS Test Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.9
module load cuda/11.8

# Check environment
echo "=== Environment Check ==="
which python
python --version
nvidia-smi
echo "========================"

cd $SLURM_SUBMIT_DIR

# Quick dependency check
python -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch not installed - will install in main job')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers not installed - will install in main job')
"

# Test basic imports from our code
echo "=== Testing MFS imports ==="
python -c "
import sys
sys.path.append('.')
try:
    from models.mfs_layer import MFSLayer
    print('✓ MFS Layer import successful')
except Exception as e:
    print(f'✗ MFS Layer import failed: {e}')

try:
    from models.full_model import MFSTransformer
    print('✓ MFS Transformer import successful')
except Exception as e:
    print(f'✗ MFS Transformer import failed: {e}')

try:
    from training.trainer import MFSTrainer
    print('✓ MFS Trainer import successful')
except Exception as e:
    print(f'✗ MFS Trainer import failed: {e}')
"

echo "=== Test completed ==="
echo "End Time: $(date)"
echo "If all imports work, you can run the full training job!" 