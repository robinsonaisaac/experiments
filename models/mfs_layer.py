import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from .safety_features import SafetyFeatureBank
from .aggregators import ErrorCorrectingAggregator


class MFSLayer(nn.Module):
    """
    Massive Feature Superposition Layer
    
    This layer embeds millions of distributed safety features using superposition
    to make safety removal computationally hard.
    """
    
    def __init__(
        self,
        d_model: int,
        n_safety_features: int = 1_000_000,
        feature_dim: int = 64,
        aggregation_type: str = "error_correcting",
        sparsity_level: float = 0.01,
        layer_idx: int = 0,
        use_efficient_computation: bool = True,
        chunk_size: int = 10000,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_safety_features = n_safety_features
        self.feature_dim = feature_dim
        self.sparsity_level = sparsity_level
        self.layer_idx = layer_idx
        self.use_efficient_computation = use_efficient_computation
        self.chunk_size = chunk_size
        
        # Initialize safety feature bank
        self.safety_features = SafetyFeatureBank(
            d_model=d_model,
            n_features=n_safety_features,
            feature_dim=feature_dim,
            sparsity_level=sparsity_level,
            use_efficient_computation=use_efficient_computation,
            chunk_size=chunk_size
        )
        
        # Initialize aggregator
        if aggregation_type == "error_correcting":
            self.aggregator = ErrorCorrectingAggregator(
                n_features=n_safety_features,
                feature_dim=feature_dim,
                redundancy_factor=3  # Reed-Solomon-like redundancy
            )
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
        
        # Safety detection threshold (learnable)
        self.safety_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Feature mixing matrix for superposition
        self.feature_mixer = nn.Linear(feature_dim, d_model, bias=False)
        
        # Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_safety_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MFS layer
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len]
            return_safety_scores: Whether to return safety scores
            
        Returns:
            modified_hidden_states: [batch_size, seq_len, d_model]
            safety_scores: [batch_size] if return_safety_scores else None
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Compute safety feature activations
        feature_activations = self.safety_features(hidden_states, attention_mask)
        # feature_activations: [batch_size, seq_len, n_safety_features]
        
        # Aggregate features using error-correcting aggregation
        aggregated_features, safety_scores = self.aggregator(
            feature_activations, 
            return_scores=True
        )
        # aggregated_features: [batch_size, seq_len, feature_dim]
        # safety_scores: [batch_size]
        
        # Mix aggregated features back into hidden states
        feature_contribution = self.feature_mixer(aggregated_features)
        
        # Apply safety gating based on aggregated scores
        # Higher safety scores = more intervention
        safety_gates = torch.sigmoid(safety_scores - self.safety_threshold)
        safety_gates = safety_gates.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
        
        # Combine original hidden states with safety features
        # Using residual connection with safety gating
        modified_hidden_states = hidden_states + safety_gates * feature_contribution
        
        # Normalize and apply dropout
        modified_hidden_states = self.layer_norm(modified_hidden_states)
        modified_hidden_states = self.dropout(modified_hidden_states)
        
        if return_safety_scores:
            return modified_hidden_states, safety_scores
        else:
            return modified_hidden_states, None
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get importance scores for all safety features"""
        return self.safety_features.get_feature_importance()
    
    def ablate_features(self, feature_indices: torch.Tensor):
        """Ablate specific features (for testing robustness)"""
        self.safety_features.ablate_features(feature_indices)
    
    def get_computational_cost(self) -> dict:
        """Return computational cost breakdown"""
        return {
            'n_safety_features': self.n_safety_features,
            'feature_dim': self.feature_dim,
            'memory_mb': self.estimate_memory_usage(),
            'flops_per_token': self.estimate_flops_per_token()
        }
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        feature_params = self.n_safety_features * self.feature_dim * 4  # 4 bytes per float32
        other_params = sum(p.numel() * 4 for p in self.parameters() 
                          if p not in self.safety_features.parameters())
        return (feature_params + other_params) / (1024 * 1024)
    
    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs per token"""
        # Simplified FLOP counting
        feature_computation = self.n_safety_features * self.d_model * 2  # multiply-add
        aggregation_flops = self.aggregator.estimate_flops()
        mixing_flops = self.feature_dim * self.d_model * 2
        return feature_computation + aggregation_flops + mixing_flops


class EfficientMFSLayer(MFSLayer):
    """
    Memory-efficient version of MFS layer for very large feature counts
    """
    
    def __init__(self, *args, **kwargs):
        # Force efficient computation
        kwargs['use_efficient_computation'] = True
        super().__init__(*args, **kwargs)
        
        # Additional optimizations
        self.use_gradient_checkpointing = True
        self.feature_cache = {}
        
    def forward(self, hidden_states, attention_mask=None, return_safety_scores=False):
        if self.use_gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(
                hidden_states, attention_mask, return_safety_scores
            )
        else:
            return super().forward(hidden_states, attention_mask, return_safety_scores)
    
    def _forward_with_checkpointing(self, hidden_states, attention_mask, return_safety_scores):
        """Forward pass with gradient checkpointing to save memory"""
        from torch.utils.checkpoint import checkpoint
        
        def forward_chunk(chunk_hidden_states):
            return super(EfficientMFSLayer, self).forward(
                chunk_hidden_states, attention_mask, return_safety_scores
            )
        
        # Process in chunks to save memory
        chunk_size = min(hidden_states.size(1), 512)  # Sequence length chunks
        chunks = torch.split(hidden_states, chunk_size, dim=1)
        
        processed_chunks = []
        all_safety_scores = []
        
        for chunk in chunks:
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :chunk.size(1)]
            else:
                chunk_mask = None
                
            chunk_output, chunk_scores = checkpoint(
                lambda x: super(EfficientMFSLayer, self).forward(
                    x, chunk_mask, return_safety_scores
                ),
                chunk
            )
            
            processed_chunks.append(chunk_output)
            if chunk_scores is not None:
                all_safety_scores.append(chunk_scores)
        
        # Concatenate results
        final_hidden_states = torch.cat(processed_chunks, dim=1)
        
        if return_safety_scores and all_safety_scores:
            # Average safety scores across chunks
            final_safety_scores = torch.stack(all_safety_scores).mean(dim=0)
            return final_hidden_states, final_safety_scores
        
        return final_hidden_states, None 