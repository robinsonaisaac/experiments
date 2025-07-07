import os
import torch
import torch.distributed as dist
from accelerate import Accelerator
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def setup_distributed_training(
    backend: str = "nccl",
    timeout_minutes: int = 30,
    mixed_precision: str = "no",
    gradient_accumulation_steps: int = 1,
    **kwargs
) -> Accelerator:
    """
    Setup distributed training with Accelerate
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
        timeout_minutes: Timeout for distributed operations
        mixed_precision: Mixed precision mode ('no', 'fp16', 'bf16')
        gradient_accumulation_steps: Gradient accumulation steps
        **kwargs: Additional arguments for Accelerator
    
    Returns:
        Accelerator instance
    """
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **kwargs
    )
    
    # Log distributed setup info
    if accelerator.is_main_process:
        logger.info(f"Distributed training setup:")
        logger.info(f"  - World size: {accelerator.num_processes}")
        logger.info(f"  - Local rank: {accelerator.local_process_index}")
        logger.info(f"  - Device: {accelerator.device}")
        logger.info(f"  - Mixed precision: {mixed_precision}")
        logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    
    return accelerator


def setup_for_colab(
    use_wandb: bool = True,
    wandb_project: str = "mfs-safety-colab",
    mixed_precision: str = "fp16",
    **kwargs
) -> Accelerator:
    """
    Setup for Google Colab training
    
    Args:
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        mixed_precision: Mixed precision mode
        **kwargs: Additional arguments
    
    Returns:
        Accelerator instance configured for Colab
    """
    
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
        logger.info("Detected Google Colab environment")
    except ImportError:
        in_colab = False
        logger.info("Not running in Google Colab")
    
    # Configure for Colab
    if in_colab:
        # Mount Google Drive if needed
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted")
        except Exception as e:
            logger.warning(f"Failed to mount Google Drive: {e}")
        
        # Set up W&B in Colab
        if use_wandb:
            try:
                import wandb
                # W&B login handled by user or environment
                logger.info("W&B available for logging")
            except ImportError:
                logger.warning("W&B not available, install with: !pip install wandb")
                use_wandb = False
    
    # Create accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="wandb" if use_wandb else None,
        project_dir="/content/drive/MyDrive/mfs_outputs" if in_colab else "./outputs",
        **kwargs
    )
    
    # Initialize W&B if requested
    if use_wandb and accelerator.is_main_process:
        try:
            import wandb
            if not wandb.run:
                wandb.init(project=wandb_project)
                logger.info(f"Initialized W&B project: {wandb_project}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    return accelerator


def optimize_for_a100(
    model: torch.nn.Module,
    mixed_precision: str = "bf16",
    compile_model: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Optimize model for A100 GPU
    
    Args:
        model: Model to optimize
        mixed_precision: Mixed precision mode
        compile_model: Whether to compile model (PyTorch 2.0+)
        **kwargs: Additional optimization arguments
    
    Returns:
        Dictionary with optimization info
    """
    
    optimizations = {}
    
    # Check if A100 is available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name:
            logger.info(f"Detected A100 GPU: {gpu_name}")
            optimizations['gpu'] = gpu_name
            
            # Enable tensor cores for A100
            if mixed_precision in ["fp16", "bf16"]:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                optimizations['tensor_cores'] = True
                logger.info("Enabled tensor cores for A100")
        else:
            logger.info(f"GPU detected but not A100: {gpu_name}")
            optimizations['gpu'] = gpu_name
    else:
        logger.warning("No CUDA GPU detected")
        optimizations['gpu'] = "cpu"
    
    # Compile model if requested and supported
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            optimizations['compiled'] = True
            logger.info("Model compiled with PyTorch 2.0+")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
            optimizations['compiled'] = False
    else:
        optimizations['compiled'] = False
    
    # Memory optimization
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get memory info
        memory_info = torch.cuda.memory_stats()
        optimizations['memory_allocated_gb'] = memory_info.get('allocated_bytes.all.current', 0) / 1e9
        optimizations['memory_reserved_gb'] = memory_info.get('reserved_bytes.all.current', 0) / 1e9
        
        logger.info(f"GPU memory - Allocated: {optimizations['memory_allocated_gb']:.2f}GB, "
                   f"Reserved: {optimizations['memory_reserved_gb']:.2f}GB")
    
    return optimizations


def estimate_memory_requirements(
    model: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    mixed_precision: str = "fp16"
) -> Dict[str, float]:
    """
    Estimate memory requirements for training
    
    Args:
        model: Model to analyze
        batch_size: Training batch size
        sequence_length: Sequence length
        mixed_precision: Mixed precision mode
    
    Returns:
        Dictionary with memory estimates (in GB)
    """
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    # Bytes per parameter based on precision
    if mixed_precision == "fp16":
        bytes_per_param = 2
    elif mixed_precision == "bf16":
        bytes_per_param = 2
    else:  # fp32
        bytes_per_param = 4
    
    # Model weights
    model_memory = param_count * bytes_per_param
    
    # Gradients (same size as model)
    gradient_memory = model_memory
    
    # Optimizer states (Adam: 2x model size)
    optimizer_memory = model_memory * 2
    
    # Activations (rough estimate)
    # Assuming transformer: batch_size * seq_len * hidden_size * num_layers * expansion_factor
    if hasattr(model, 'config'):
        hidden_size = getattr(model.config, 'd_model', getattr(model.config, 'hidden_size', 768))
        num_layers = getattr(model.config, 'n_layers', getattr(model.config, 'num_hidden_layers', 12))
    else:
        hidden_size = 768  # Default
        num_layers = 12    # Default
    
    # Rough activation memory estimate
    activation_memory = batch_size * sequence_length * hidden_size * num_layers * 4 * bytes_per_param
    
    # Total memory
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'model_gb': model_memory / 1e9,
        'gradients_gb': gradient_memory / 1e9,
        'optimizer_gb': optimizer_memory / 1e9,
        'activations_gb': activation_memory / 1e9,
        'total_gb': total_memory / 1e9,
        'param_count': param_count,
    }


def check_gpu_compatibility(mixed_precision: str = "fp16") -> Dict[str, Any]:
    """
    Check GPU compatibility for training
    
    Args:
        mixed_precision: Mixed precision mode to check
    
    Returns:
        Dictionary with compatibility info
    """
    
    compatibility = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mixed_precision_supported': False,
        'tensor_cores_available': False,
        'recommended_settings': {}
    }
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compatibility['gpu_name'] = gpu_name
        
        # Check compute capability
        capability = torch.cuda.get_device_capability(0)
        compatibility['compute_capability'] = f"{capability[0]}.{capability[1]}"
        
        # Check mixed precision support
        if capability[0] >= 7:  # Volta and newer
            compatibility['mixed_precision_supported'] = True
            if mixed_precision == "bf16" and capability[0] >= 8:  # Ampere and newer
                compatibility['bf16_supported'] = True
            else:
                compatibility['bf16_supported'] = False
        
        # Check for tensor cores
        if capability[0] >= 7:
            compatibility['tensor_cores_available'] = True
        
        # Memory info
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        compatibility['memory_gb'] = memory_gb
        
        # Recommendations
        recommendations = {}
        if "A100" in gpu_name:
            recommendations['mixed_precision'] = 'bf16'
            recommendations['batch_size'] = 'large (16-32)'
            recommendations['compile'] = True
        elif "V100" in gpu_name:
            recommendations['mixed_precision'] = 'fp16'
            recommendations['batch_size'] = 'medium (8-16)'
            recommendations['compile'] = False
        elif "T4" in gpu_name:
            recommendations['mixed_precision'] = 'fp16'
            recommendations['batch_size'] = 'small (2-8)'
            recommendations['compile'] = False
        else:
            recommendations['mixed_precision'] = 'fp16' if compatibility['mixed_precision_supported'] else 'no'
            recommendations['batch_size'] = 'adjust based on memory'
            recommendations['compile'] = True if capability[0] >= 8 else False
        
        compatibility['recommended_settings'] = recommendations
    
    return compatibility


def log_system_info():
    """Log system information for debugging"""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # Check for Google Colab
    try:
        import google.colab
        logger.info("Running in Google Colab")
    except ImportError:
        logger.info("Not running in Google Colab")
    
    logger.info("=== End System Information ===")


# Auto-detect and configure based on environment
def auto_configure_training(**kwargs) -> Accelerator:
    """
    Auto-configure training based on detected environment
    """
    
    log_system_info()
    
    # Check environment
    try:
        import google.colab
        logger.info("Auto-configuring for Google Colab")
        return setup_for_colab(**kwargs)
    except ImportError:
        pass
    
    # Check for A100 and optimize
    compatibility = check_gpu_compatibility()
    if "A100" in compatibility.get('gpu_name', ''):
        logger.info("Auto-configuring for A100")
        mixed_precision = "bf16"
    elif compatibility.get('mixed_precision_supported', False):
        logger.info("Auto-configuring with mixed precision")
        mixed_precision = "fp16"
    else:
        logger.info("Auto-configuring without mixed precision")
        mixed_precision = "no"
    
    return setup_distributed_training(
        mixed_precision=mixed_precision,
        **kwargs
    ) 