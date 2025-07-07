# Running MFS on Oxford HTC Cluster

This guide explains how to run the Massive Feature Superposition (MFS) safety experiments on Oxford University's HTC cluster.

## 🚀 Quick Start

### 1. Initial Setup (Run Once)

After logging into the HTC cluster:

```bash
cd ~/CryptoSafety
bash cluster_setup.sh
source ~/.bashrc
```

### 2. Test Your Setup

Submit a quick test job to verify everything works:

```bash
sbatch mfs_training_test.slurm
```

Check job status:
```bash
squeue -u $USER
```

### 3. Run Full Experiments

Once the test passes, submit the full training:

```bash
sbatch mfs_training.slurm
```

## 📋 Available SLURM Scripts

### `mfs_training_test.slurm`
- **Purpose**: Quick validation test (10 minutes)
- **Resources**: 1 GPU, 2 CPUs, 8GB RAM
- **What it does**: Tests environment setup and imports

### `mfs_training.slurm`
- **Purpose**: Full MFS training experiments (up to 12 hours)
- **Resources**: 1 GPU, 8 CPUs, 64GB RAM  
- **What it does**: 
  - Automatically detects GPU memory and runs appropriate scale
  - Installs dependencies in virtual environment
  - Runs training, scaling experiments, and attack resistance tests
  - Generates analysis reports

## 🎛️ Configuration Options

The main script automatically adapts based on available GPU memory:

| GPU Memory | Experiment Scale | Features | Expected Runtime |
|------------|------------------|----------|------------------|
| >30GB (A100/V100-32GB) | Large | 1M+ | 8-12 hours |
| >15GB (RTX8000/V100-16GB) | Medium | 100K | 4-6 hours |
| <15GB (smaller GPUs) | Small | 10K | 1-2 hours |

## 🔧 Customizing Experiments

### Modify GPU Requirements

Edit the SLURM headers in the scripts:

```bash
#SBATCH --gres=gpu:1              # Request any GPU
#SBATCH --gres=gpu:a100:1         # Request specific A100
#SBATCH --gres=gpu:v100:1         # Request specific V100
```

### Change Time Limits

```bash
#SBATCH --partition=short         # Max 12 hours
#SBATCH --partition=medium        # Max 48 hours  
#SBATCH --partition=long          # No limit (lower priority)
```

### Request More Resources

```bash
#SBATCH --cpus-per-task=16        # More CPUs
#SBATCH --mem=128G                # More memory
#SBATCH --nodes=2                 # Multiple nodes (for multi-GPU)
```

## 📊 Monitoring Your Jobs

### Check Job Status
```bash
squeue -u $USER                   # Your jobs
squeue -p short                   # All short partition jobs
```

### Check GPU Availability
```bash
gpu-status                        # Custom alias for GPU nodes
sinfo -p gpu,contrib-gpu          # Raw SLURM info
```

### View Job Output
```bash
tail -f logs/mfs_training_JOBID.out     # Follow output log
tail -f logs/mfs_training_JOBID.err     # Follow error log
```

### Connect to Running Job
```bash
srun --jobid=JOBID --pty /bin/bash      # Interactive shell on compute node
```

## 📁 Output Structure

After completion, you'll find:

```
outputs/
├── mfs_large_JOBID/              # Main training results
│   ├── checkpoints/
│   ├── final_model/
│   └── training_logs/
├── scaling_JOBID/                # Scaling law experiments  
├── attacks_JOBID/                # Attack resistance results
└── mfs_analysis_JOBID.html       # Combined analysis report
```

## 🐛 Troubleshooting

### Common Issues

**Job fails immediately:**
```bash
# Check if test job worked first
sbatch mfs_training_test.slurm

# Check for typos in script
cat mfs_training.slurm | head -20
```

**Out of memory errors:**
```bash
# Reduce batch size or feature count
# Or request more memory: #SBATCH --mem=128G
```

**CUDA errors:**
```bash
# Check GPU allocation
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Verify CUDA module loaded
module list
```

**Python import errors:**
```bash
# Check if CryptoSafety directory exists and has correct structure
ls -la ~/CryptoSafety/models/
ls -la ~/CryptoSafety/training/
```

### Getting Help

1. **Check logs first**: `tail logs/mfs_training_JOBID.err`
2. **Oxford HTC docs**: [ARC User Guide](https://arc-user-guide.readthedocs.io/)
3. **Contact support**: `support@arc.ox.ac.uk`

## ⚡ Performance Tips

### Optimize for Oxford HTC

1. **Use fast storage**: Write outputs to local `/tmp` during job, copy to home at end
2. **Batch multiple experiments**: Use job arrays for parameter sweeps
3. **Use appropriate partitions**: `short` for <12hrs, `medium` for <48hrs
4. **Monitor resource usage**: Don't over-request CPUs/memory

### Example Job Array

```bash
#!/bin/bash
#SBATCH --job-name=mfs-sweep
#SBATCH --array=1-10
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Run different feature counts
FEATURES=(10000 50000 100000 500000 1000000)
N_FEATURES=${FEATURES[$SLURM_ARRAY_TASK_ID-1]}

python experiments/train_scaling.py --n_safety_features $N_FEATURES
```

## 🎯 Expected Results

### Successful Run Indicators

- ✅ No CUDA errors in logs
- ✅ Training loss decreases over time  
- ✅ Safety scores >0.8 for harmful examples
- ✅ Attack resistance >90% after feature ablation
- ✅ Analysis report generated with scaling plots

### Key Metrics to Check

1. **GPU Utilization**: >80% during training
2. **Memory Usage**: <90% of allocated
3. **Training Speed**: ~X steps/second (depends on scale)
4. **Safety Performance**: High precision/recall on safety classification
5. **Attack Resistance**: Computational cost scaling superlinearly

## 📚 Further Reading

- [MFS Paper](../README.md): Core research concepts
- [Implementation Details](../IMPLEMENTATION_SUMMARY.md): Technical architecture
- [Oxford ARC Documentation](https://arc-user-guide.readthedocs.io/): Cluster-specific guides
- [SLURM Reference](https://slurm.schedmd.com/documentation.html): Complete SLURM docs 