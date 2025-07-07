#!/usr/bin/env python3
"""
Local training script for testing MFS on tiny models
"""

import os
import sys
import logging
import torch
from transformers import AutoTokenizer
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.full_model import MFSTransformer, MFSTransformerConfig
from training.trainer import MFSTrainer
from training.datasets import create_tiny_dataset
from training.losses import MFSLoss
from training.distributed import auto_configure_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function for local testing"""
    
    logger.info("Starting MFS local training experiment")
    
    # Configuration for tiny model
    config = MFSTransformerConfig(
        base_model_name="gpt2",  # Use tiny GPT-2
        d_model=768,
        n_layers=12,
        n_safety_features=10_000,  # Much smaller for local testing
        feature_dim=32,  # Smaller feature dimension
        mfs_layers=[0, 3, 6, 9],  # Add MFS to 4 layers
        sparsity_level=0.05,  # Higher sparsity for efficiency
        aggregation_type="error_correcting",
        use_efficient_computation=True,
        chunk_size=1000,  # Smaller chunks
        safety_alpha=1.0,
    )
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create tiny dataset for testing
    logger.info("Creating tiny dataset...")
    train_dataset = create_tiny_dataset(tokenizer, size=200)  # Very small dataset
    eval_dataset = create_tiny_dataset(tokenizer, size=50)
    
    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Eval dataset: {len(eval_dataset)} examples")
    logger.info(f"Safety distribution: {train_dataset.get_safety_distribution()}")
    
    # Initialize model
    logger.info("Initializing MFS model...")
    model = MFSTransformer(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    mfs_params = sum(p.numel() for p in model.mfs_layers.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"MFS parameters: {mfs_params:,} ({mfs_params/total_params*100:.1f}%)")
    
    # Get computational cost estimate
    cost_info = model.get_computational_cost()
    logger.info(f"Estimated memory usage: {cost_info['memory_mb']:.1f} MB")
    logger.info(f"MFS layers: {cost_info['mfs_layers_count']}")
    
    # Initialize loss function
    loss_fn = MFSLoss(
        lm_weight=1.0,
        safety_weight=2.0,  # Higher weight for safety
        diversity_weight=0.1,
        robustness_weight=0.05,  # Lower for tiny model
        sparsity_weight=0.01,
        adaptive_weights=True,
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = MFSTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_fn=loss_fn,
        learning_rate=5e-4,  # Higher learning rate for quick testing  
        weight_decay=0.01,
        warmup_steps=50,
        max_steps=500,  # Short training for testing
        eval_steps=100,
        save_steps=200,
        logging_steps=25,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        batch_size=2,  # Small batch size for local testing
        eval_batch_size=2,
        output_dir="./outputs/tiny_local_test",
        experiment_name="mfs_tiny_local",
        use_wandb=False,  # Disable W&B for local testing
        safety_loss_weight=2.0,
        diversity_loss_weight=0.1,
        robustness_loss_weight=0.05,
        feature_sparsity_weight=0.01,
    )
    
    # Test forward pass
    logger.info("Testing forward pass...")
    try:
        sample_batch = next(iter(trainer.train_dataloader))
        with torch.no_grad():
            outputs = model(**sample_batch, return_safety_scores=True)
            logger.info("Forward pass successful!")
            logger.info(f"Output keys: {list(outputs.keys())}")
            if 'safety_scores' in outputs:
                logger.info(f"Safety scores shape: {outputs['safety_scores'].shape}")
                logger.info(f"Safety scores range: [{outputs['safety_scores'].min():.3f}, {outputs['safety_scores'].max():.3f}]")
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return
    
    # Test loss computation
    logger.info("Testing loss computation...")
    try:
        loss_dict = loss_fn(outputs=outputs, batch=sample_batch, model=model)
        logger.info("Loss computation successful!")
        logger.info(f"Loss components: {loss_dict}")
    except Exception as e:
        logger.error(f"Loss computation failed: {e}")
        return
    
    # Start training
    logger.info("Starting training...")
    try:
        training_summary = trainer.train()
        logger.info("Training completed successfully!")
        logger.info(f"Training summary: {training_summary}")
        
        # Test safety evaluation
        logger.info("Testing safety evaluation...")
        resistance_info = model.measure_attack_resistance(attack_budget=100)
        logger.info(f"Attack resistance: {resistance_info}")
        
        safety_stats = model.get_safety_statistics()
        logger.info(f"Safety statistics: {safety_stats}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test generation
    logger.info("Testing safe generation...")
    try:
        model.eval()
        test_prompts = [
            "How can I help",
            "Tell me how to",
            "I want to know",
        ]
        
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            generation_result = model.generate_safe(
                input_ids,
                max_length=input_ids.size(1) + 20,
                safety_threshold=0.7,
                do_sample=True,
                temperature=0.8,
            )
            
            generated_text = tokenizer.decode(generation_result['generated_ids'][0], skip_special_tokens=True)
            logger.info(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")
            logger.info(f"Generation stopped by safety: {generation_result['generation_stopped_by_safety']}")
            
    except Exception as e:
        logger.error(f"Generation test failed: {e}")
    
    logger.info("Local testing completed successfully!")


if __name__ == "__main__":
    main() 