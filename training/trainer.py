import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_scheduler
from accelerate import Accelerator
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import time
import os
import json
from pathlib import Path

from .losses import MFSLoss
from .datasets import MFSDataset
from models.full_model import MFSTransformer, MFSTransformerConfig


logger = logging.getLogger(__name__)


class MFSTrainer:
    """
    Multi-objective trainer for MFS-enhanced models
    """
    
    def __init__(
        self,
        model: MFSTransformer,
        train_dataset: MFSDataset,
        eval_dataset: Optional[MFSDataset] = None,
        loss_fn: Optional[MFSLoss] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        bf16: bool = False,
        dataloader_num_workers: int = 0,
        batch_size: int = 4,
        eval_batch_size: Optional[int] = None,
        output_dir: str = "./outputs",
        experiment_name: str = "mfs_experiment",
        use_wandb: bool = True,
        wandb_project: str = "mfs-safety",
        safety_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.1,
        robustness_loss_weight: float = 0.1,
        feature_sparsity_weight: float = 0.01,
        scheduler_type: str = "cosine",
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        
        # Loss weights
        self.safety_loss_weight = safety_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.robustness_loss_weight = robustness_loss_weight
        self.feature_sparsity_weight = feature_sparsity_weight
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision='fp16' if fp16 else ('bf16' if bf16 else 'no'),
            log_with="wandb" if use_wandb else None,
            project_dir=output_dir,
        )
        
        # Initialize loss function
        if loss_fn is None:
            self.loss_fn = MFSLoss(
                safety_weight=safety_loss_weight,
                diversity_weight=diversity_loss_weight,
                robustness_weight=robustness_loss_weight,
                sparsity_weight=feature_sparsity_weight
            )
        else:
            self.loss_fn = loss_fn
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Initialize scheduler
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=learning_rate * 0.1
            )
        elif scheduler_type == "linear":
            self.scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps
            )
        else:
            self.scheduler = None
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
        )
        
        self.eval_dataloader = None
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=dataloader_num_workers,
                collate_fn=eval_dataset.collate_fn if hasattr(eval_dataset, 'collate_fn') else None
            )
        
        # Prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.scheduler,
        )
        
        # Logging and checkpointing
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Initialize wandb
        if use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config={
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'max_steps': max_steps,
                    'model_config': model.config.__dict__ if hasattr(model.config, '__dict__') else {},
                    'safety_loss_weight': safety_loss_weight,
                    'diversity_loss_weight': diversity_loss_weight,
                    'robustness_loss_weight': robustness_loss_weight,
                    **kwargs
                }
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.training_history = []
        
        # Computational cost tracking
        self.total_flops = 0
        self.total_tokens_processed = 0
        
        logger.info(f"Initialized MFS trainer with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        """
        logger.info(f"Starting training for {self.max_steps} steps")
        
        self.model.train()
        train_iterator = iter(self.train_dataloader)
        
        # Training metrics
        total_loss = 0.0
        total_lm_loss = 0.0
        total_safety_loss = 0.0
        total_diversity_loss = 0.0
        total_robustness_loss = 0.0
        
        start_time = time.time()
        
        progress_bar = tqdm(
            range(self.max_steps),
            desc="Training",
            disable=not self.accelerator.is_main_process
        )
        
        for step in progress_bar:
            self.global_step = step
            
            # Get next batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
                self.epoch += 1
            
            # Forward pass
            with self.accelerator.accumulate(self.model):
                # Model forward pass
                outputs = self.model(**batch, return_safety_scores=True)
                
                # Compute multi-objective loss
                loss_dict = self.loss_fn(
                    outputs=outputs,
                    batch=batch,
                    model=self.model,
                    global_step=self.global_step
                )
                
                loss = loss_dict['total_loss']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_lm_loss += loss_dict.get('lm_loss', 0.0)
            total_safety_loss += loss_dict.get('safety_loss', 0.0)
            total_diversity_loss += loss_dict.get('diversity_loss', 0.0)
            total_robustness_loss += loss_dict.get('robustness_loss', 0.0)
            
            # Update computational cost tracking
            batch_size = batch['input_ids'].size(0)
            seq_len = batch['input_ids'].size(1)
            self.total_tokens_processed += batch_size * seq_len
            
            # Logging
            if step % self.logging_steps == 0:
                self._log_training_metrics(
                    step, 
                    {
                        'total_loss': total_loss / (step + 1),
                        'lm_loss': total_lm_loss / (step + 1),
                        'safety_loss': total_safety_loss / (step + 1),
                        'diversity_loss': total_diversity_loss / (step + 1),
                        'robustness_loss': total_robustness_loss / (step + 1),
                        'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate,
                        'epoch': self.epoch,
                    }
                )
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss / (step + 1):.4f}",
                    'safety': f"{total_safety_loss / (step + 1):.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate:.2e}"
                })
            
            # Evaluation
            if step % self.eval_steps == 0 and step > 0:
                eval_metrics = self._evaluate()
                self._log_eval_metrics(step, eval_metrics)
                
                # Save best model
                if eval_metrics.get('eval_loss', float('inf')) < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics['eval_loss']
                    self._save_checkpoint(step, is_best=True)
                
                self.model.train()  # Return to training mode
            
            # Save checkpoint
            if step % self.save_steps == 0 and step > 0:
                self._save_checkpoint(step, is_best=False)
            
            # Early stopping based on computational cost
            if self._should_early_stop():
                logger.info(f"Early stopping at step {step}")
                break
        
        # Final evaluation and save
        if self.eval_dataloader is not None:
            final_eval_metrics = self._evaluate()
            self._log_eval_metrics(self.global_step, final_eval_metrics)
        
        self._save_checkpoint(self.global_step, is_best=False, is_final=True)
        
        # Training summary
        training_time = time.time() - start_time
        training_summary = {
            'total_steps': self.global_step,
            'total_epochs': self.epoch,
            'training_time_hours': training_time / 3600,
            'tokens_per_second': self.total_tokens_processed / training_time,
            'final_loss': total_loss / self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'computational_cost': self.model.get_computational_cost(),
        }
        
        logger.info(f"Training completed: {training_summary}")
        return training_summary
    
    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluation loop
        """
        if self.eval_dataloader is None:
            return {}
        
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_safety_loss = 0.0
        total_safety_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                outputs = self.model(**batch, return_safety_scores=True)
                
                loss_dict = self.loss_fn(
                    outputs=outputs,
                    batch=batch,
                    model=self.model,
                    global_step=self.global_step
                )
                
                total_loss += loss_dict['total_loss'].item()
                total_lm_loss += loss_dict.get('lm_loss', 0.0)
                total_safety_loss += loss_dict.get('safety_loss', 0.0)
                
                # Compute safety accuracy
                if 'safety_labels' in batch and outputs.get('safety_scores') is not None:
                    safety_preds = (outputs['safety_scores'] > 0.5).float()
                    safety_acc = (safety_preds == batch['safety_labels'].float()).float().mean()
                    total_safety_accuracy += safety_acc.item()
                
                num_batches += 1
        
        eval_metrics = {
            'eval_loss': total_loss / num_batches,
            'eval_lm_loss': total_lm_loss / num_batches,
            'eval_safety_loss': total_safety_loss / num_batches,
            'eval_safety_accuracy': total_safety_accuracy / num_batches,
        }
        
        # Add computational resistance metrics
        resistance_metrics = self.model.measure_attack_resistance(attack_budget=1000)
        eval_metrics.update({f'eval_{k}': v for k, v in resistance_metrics.items()})
        
        return eval_metrics
    
    def _log_training_metrics(self, step: int, metrics: Dict[str, float]):
        """Log training metrics"""
        if self.accelerator.is_main_process:
            # Console logging
            logger.info(f"Step {step}: {metrics}")
            
            # Wandb logging
            if wandb.run is not None:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            
            # Store in history
            self.training_history.append({'step': step, **metrics})
    
    def _log_eval_metrics(self, step: int, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        if self.accelerator.is_main_process:
            logger.info(f"Eval at step {step}: {metrics}")
            
            if wandb.run is not None:
                wandb.log(metrics, step=step)
    
    def _save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        # Prepare checkpoint
        checkpoint = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_eval_loss': self.best_eval_loss,
            'training_history': self.training_history,
            'config': self.model.config.__dict__ if hasattr(self.model.config, '__dict__') else {},
        }
        
        # Save paths
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at step {step} with eval loss {self.best_eval_loss:.4f}")
        
        if is_final:
            final_path = checkpoint_dir / "final_model.pt"
            torch.save(checkpoint, final_path)
            logger.info(f"Saved final model at step {step}")
        
        # Save model config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(checkpoint['config'], f, indent=2)
        
        logger.info(f"Saved checkpoint at {checkpoint_path}")
    
    def _should_early_stop(self) -> bool:
        """Check if training should stop early based on computational limits"""
        # Simple heuristic: stop if we're using too much memory
        cost_info = self.model.get_computational_cost()
        if cost_info.get('memory_mb', 0) > 32000:  # 32GB limit
            logger.warning("Memory usage too high, stopping early")
            return True
        
        return False
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Loaded checkpoint from step {self.global_step}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        cost_info = self.model.get_computational_cost()
        safety_stats = self.model.get_safety_statistics()
        
        return {
            'training_steps': self.global_step,
            'epochs': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'computational_cost': cost_info,
            'safety_statistics': safety_stats,
            'model_config': self.model.config.__dict__ if hasattr(self.model.config, '__dict__') else {},
            'training_history': self.training_history[-10:],  # Last 10 entries
        } 