#!/bin/bash
#SBATCH --job-name=mfs-safety-training
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --output=logs/mfs_training_%j.out
#SBATCH --error=logs/mfs_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=univ5678@ox.ac.uk

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Check GPU availability
echo "=== GPU Information ==="
nvidia-smi
echo "========================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the job submission directory
cd $SLURM_SUBMIT_DIR

# Create Python virtual environment if it doesn't exist
if [ ! -d "mfs_env" ]; then
    echo "Creating Python virtual environment..."
    python -m venv mfs_env
fi

# Activate virtual environment
source mfs_env/bin/activate

# Install/upgrade required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate wandb tensorboard
pip install numpy scipy scikit-learn matplotlib seaborn
pip install -r requirements.txt

# Verify PyTorch CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Configuration selection based on available GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU Memory: ${GPU_MEM}MB"

if [ "$GPU_MEM" -gt 30000 ]; then
    # High memory GPU (A100, V100 32GB+) - run large experiment
    echo "Running large-scale MFS experiment (1M+ features)..."
    python experiments/train_large_scale.py \
        --config large \
        --n_safety_features 1000000 \
        --feature_dim 128 \
        --batch_size 4 \
        --max_steps 5000 \
        --eval_steps 500 \
        --save_steps 1000 \
        --learning_rate 3e-4 \
        --warmup_steps 200 \
        --output_dir "./outputs/mfs_large_$SLURM_JOB_ID" \
        --experiment_name "mfs_large_htc_$SLURM_JOB_ID" \
        --use_wandb true \
        --safety_loss_weight 2.0 \
        --diversity_loss_weight 0.1 \
        --robustness_loss_weight 0.1 \
        --gradient_accumulation_steps 4
elif [ "$GPU_MEM" -gt 15000 ]; then
    # Medium memory GPU (RTX8000, V100 16GB) - run medium experiment  
    echo "Running medium-scale MFS experiment (100K features)..."
    python experiments/train_medium_scale.py \
        --config medium \
        --n_safety_features 100000 \
        --feature_dim 64 \
        --batch_size 8 \
        --max_steps 3000 \
        --eval_steps 300 \
        --save_steps 600 \
        --learning_rate 5e-4 \
        --warmup_steps 150 \
        --output_dir "./outputs/mfs_medium_$SLURM_JOB_ID" \
        --experiment_name "mfs_medium_htc_$SLURM_JOB_ID" \
        --use_wandb true \
        --safety_loss_weight 2.0 \
        --diversity_loss_weight 0.1 \
        --robustness_loss_weight 0.05 \
        --gradient_accumulation_steps 2
else
    # Lower memory GPU - run small experiment
    echo "Running small-scale MFS experiment (10K features)..."
    python experiments/train_tiny_local.py \
        --output_dir "./outputs/mfs_small_$SLURM_JOB_ID" \
        --experiment_name "mfs_small_htc_$SLURM_JOB_ID" \
        --use_wandb true
fi

# Run scaling law experiments
echo "Running scaling law experiments..."
python experiments/scaling_experiments.py \
    --output_dir "./outputs/scaling_$SLURM_JOB_ID" \
    --feature_counts "1000,5000,10000,50000" \
    --max_steps_per_run 1000

# Run attack resistance tests
echo "Running attack resistance evaluation..."
python experiments/attack_resistance.py \
    --model_path "./outputs/mfs_*_$SLURM_JOB_ID/final_model" \
    --attack_budgets "100,500,1000,5000" \
    --output_dir "./outputs/attacks_$SLURM_JOB_ID"

# Generate final analysis report
echo "Generating analysis report..."
python experiments/analyze_results.py \
    --results_dir "./outputs" \
    --job_id "$SLURM_JOB_ID" \
    --output_file "./outputs/mfs_analysis_$SLURM_JOB_ID.html"

echo "End Time: $(date)"
echo "Job completed successfully!"

# Cleanup temporary files
echo "Cleaning up temporary files..."
rm -rf /tmp/mfs_temp_*

# Deactivate virtual environment
deactivate

echo "=== Job Summary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Elapsed Time: $SECONDS seconds"
echo "Output directory: ./outputs/"
echo "Logs: logs/mfs_training_$SLURM_JOB_ID.{out,err}"
echo "===================" 