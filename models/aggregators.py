import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import numpy as np


class ErrorCorrectingAggregator(nn.Module):
    """
    Error-correcting aggregator using Reed-Solomon-like redundancy
    for robust safety feature combination
    """
    
    def __init__(
        self,
        n_features: int,
        feature_dim: int = 64,
        redundancy_factor: int = 3,
        aggregation_method: str = "attention",
        use_consensus: bool = True,
        consensus_threshold: float = 0.7,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.feature_dim = feature_dim
        self.redundancy_factor = redundancy_factor
        self.aggregation_method = aggregation_method
        self.use_consensus = use_consensus
        self.consensus_threshold = consensus_threshold
        
        # Create redundant feature groups
        self.n_groups = n_features // redundancy_factor
        self.features_per_group = redundancy_factor
        
        # Group aggregation networks
        if aggregation_method == "attention":
            self.group_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.group_projector = nn.Linear(1, feature_dim)
        elif aggregation_method == "mlp":
            self.group_mlp = nn.Sequential(
                nn.Linear(redundancy_factor, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Final aggregation layer
        self.final_aggregator = nn.Sequential(
            nn.Linear(self.n_groups, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Safety score predictor
        self.safety_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Consensus mechanism
        if use_consensus:
            self.consensus_weights = nn.Parameter(
                torch.ones(self.n_groups) / self.n_groups
            )
        
        # Error detection network
        self.error_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        feature_activations: torch.Tensor,
        return_scores: bool = False,
        return_consensus: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Aggregate safety features with error correction
        
        Args:
            feature_activations: [batch_size, seq_len, n_features]
            return_scores: Whether to return safety scores
            return_consensus: Whether to return consensus information
            
        Returns:
            aggregated_features: [batch_size, seq_len, feature_dim]
            safety_scores: [batch_size] if return_scores else None
        """
        batch_size, seq_len, n_features = feature_activations.shape
        
        # Reshape into redundant groups
        # [batch_size, seq_len, n_groups, features_per_group]
        grouped_features = feature_activations.view(
            batch_size, seq_len, self.n_groups, self.features_per_group
        )
        
        # Aggregate within each group
        if self.aggregation_method == "attention":
            group_aggregates = self._aggregate_with_attention(grouped_features)
        elif self.aggregation_method == "mlp":
            group_aggregates = self._aggregate_with_mlp(grouped_features)
        else:
            # Simple averaging as fallback
            group_aggregates = grouped_features.mean(dim=-1)
        
        # Apply consensus mechanism
        if self.use_consensus:
            group_aggregates, consensus_info = self._apply_consensus(group_aggregates)
        
        # Detect and correct errors
        corrected_aggregates = self._error_correction(group_aggregates)
        
        # Final aggregation across groups
        # [batch_size, seq_len, n_groups] -> [batch_size, seq_len, feature_dim]
        final_features = self.final_aggregator(corrected_aggregates)
        
        # Compute safety scores if requested
        safety_scores = None
        if return_scores:
            # Pool across sequence length for safety scoring
            pooled_features = final_features.mean(dim=1)  # [batch_size, feature_dim]
            safety_scores = self.safety_predictor(pooled_features).squeeze(-1)  # [batch_size]
        
        if return_consensus and self.use_consensus:
            return final_features, safety_scores, consensus_info
        else:
            return final_features, safety_scores
    
    def _aggregate_with_attention(self, grouped_features):
        """Aggregate features within groups using attention"""
        batch_size, seq_len, n_groups, features_per_group = grouped_features.shape
        
        # Reshape for attention: [batch_size * seq_len * n_groups, features_per_group, 1]
        reshaped = grouped_features.view(-1, features_per_group, 1)
        
        # Project to feature_dim
        projected = self.group_projector(reshaped)  # [batch_size * seq_len * n_groups, features_per_group, feature_dim]
        
        # Apply self-attention within each group
        attended, _ = self.group_attention(projected, projected, projected)
        
        # Average across features_per_group dimension
        aggregated = attended.mean(dim=1)  # [batch_size * seq_len * n_groups, feature_dim]
        
        # Reshape back: [batch_size, seq_len, n_groups, feature_dim]
        return aggregated.view(batch_size, seq_len, n_groups, -1)
    
    def _aggregate_with_mlp(self, grouped_features):
        """Aggregate features within groups using MLP"""
        # Apply MLP to last dimension (features_per_group)
        return self.group_mlp(grouped_features)
    
    def _apply_consensus(self, group_aggregates):
        """Apply consensus mechanism across groups"""
        batch_size, seq_len, n_groups, feature_dim = group_aggregates.shape
        
        # Compute pairwise similarities between groups
        # Flatten for easier computation
        flat_aggregates = group_aggregates.view(batch_size * seq_len, n_groups, feature_dim)
        
        # Compute consensus scores
        consensus_scores = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                similarity = F.cosine_similarity(
                    flat_aggregates[:, i, :], 
                    flat_aggregates[:, j, :], 
                    dim=-1
                )
                consensus_scores.append(similarity)
        
        # Average consensus score
        if consensus_scores:
            avg_consensus = torch.stack(consensus_scores).mean(dim=0)
            avg_consensus = avg_consensus.view(batch_size, seq_len)
        else:
            avg_consensus = torch.ones(batch_size, seq_len, device=group_aggregates.device)
        
        # Weight groups by consensus
        consensus_weights = self.consensus_weights.softmax(dim=0)
        weighted_aggregates = group_aggregates * consensus_weights.view(1, 1, -1, 1)
        
        # Sum across groups dimension
        final_aggregates = weighted_aggregates.sum(dim=2)  # [batch_size, seq_len, feature_dim]
        
        consensus_info = {
            'consensus_scores': avg_consensus,
            'consensus_weights': consensus_weights
        }
        
        return final_aggregates, consensus_info
    
    def _error_correction(self, group_aggregates):
        """Apply error correction to group aggregates"""
        if not self.use_consensus:
            # Simple error correction without consensus
            return self._simple_error_correction(group_aggregates)
        
        # For consensus-based aggregation, group_aggregates is already processed
        return group_aggregates
    
    def _simple_error_correction(self, group_aggregates):
        """Simple error correction for non-consensus mode"""
        batch_size, seq_len, n_groups, feature_dim = group_aggregates.shape
        
        # Detect outliers in each group
        corrected_aggregates = []
        
        for group_idx in range(n_groups):
            group_features = group_aggregates[:, :, group_idx, :]  # [batch_size, seq_len, feature_dim]
            
            # Compute error probability
            error_prob = self.error_detector(group_features)  # [batch_size, seq_len, 1]
            
            # Apply correction based on error probability
            # If error probability is high, reduce the contribution of this group
            correction_factor = 1.0 - error_prob
            corrected_features = group_features * correction_factor
            
            corrected_aggregates.append(corrected_features)
        
        return torch.stack(corrected_aggregates, dim=2)  # [batch_size, seq_len, n_groups, feature_dim]
    
    def estimate_flops(self) -> int:
        """Estimate FLOPs for this aggregator"""
        # Simplified FLOP estimation
        attention_flops = 0
        if self.aggregation_method == "attention":
            attention_flops = self.n_groups * self.features_per_group * self.feature_dim * 4
        
        mlp_flops = 0
        if self.aggregation_method == "mlp":
            mlp_flops = self.n_groups * self.features_per_group * self.feature_dim * 2
        
        final_aggregation_flops = self.n_groups * self.feature_dim * 2
        
        return attention_flops + mlp_flops + final_aggregation_flops
    
    def get_redundancy_info(self) -> dict:
        """Get information about redundancy structure"""
        return {
            'n_groups': self.n_groups,
            'features_per_group': self.features_per_group,
            'redundancy_factor': self.redundancy_factor,
            'total_features': self.n_features
        }


class SimpleAggregator(nn.Module):
    """
    Simplified aggregator for testing and ablation studies
    """
    
    def __init__(self, n_features: int, feature_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.feature_dim = feature_dim
        
        # Simple linear aggregation
        self.aggregator = nn.Linear(n_features, feature_dim)
        
        # Safety scorer
        self.safety_scorer = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, feature_activations, return_scores=True):
        batch_size, seq_len, n_features = feature_activations.shape
        
        # Simple linear aggregation
        aggregated = self.aggregator(feature_activations)
        
        if return_scores:
            # Pool and compute safety scores
            pooled = aggregated.max(dim=1)[0]  # Max pooling across sequence
            safety_scores = self.safety_scorer(pooled).squeeze(-1)
            return aggregated, safety_scores
        
        return aggregated, None
    
    def estimate_flops(self):
        return self.n_features * self.feature_dim * 2


class AttentionAggregator(nn.Module):
    """
    Attention-based aggregator for more sophisticated feature combination
    """
    
    def __init__(
        self, 
        n_features: int, 
        feature_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        self.n_features = n_features
        self.feature_dim = feature_dim
        
        # Project features to feature_dim if needed
        if n_features != feature_dim:
            self.input_projection = nn.Linear(n_features, feature_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                feature_dim, num_heads, batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers)
        ])
        
        # Safety scorer
        self.safety_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, feature_activations, return_scores=True):
        # Project to feature_dim
        x = self.input_projection(feature_activations)
        
        # Apply attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            residual = x
            attended, _ = attention(x, x, x)
            x = layer_norm(attended + residual)
        
        if return_scores:
            # Pool and compute safety scores
            pooled = x.mean(dim=1)  # Mean pooling across sequence
            safety_scores = self.safety_scorer(pooled).squeeze(-1)
            return x, safety_scores
        
        return x, None
    
    def estimate_flops(self):
        flops = 0
        if isinstance(self.input_projection, nn.Linear):
            flops += self.n_features * self.feature_dim * 2
        
        # Attention FLOPs (approximate)
        for _ in self.attention_layers:
            flops += self.feature_dim * self.feature_dim * 4  # Q, K, V, output projections
        
        return flops 