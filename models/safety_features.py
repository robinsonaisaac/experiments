import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math
import numpy as np
from einops import rearrange, einsum


class SafetyFeatureBank(nn.Module):
    """
    Efficient bank of millions of safety micro-features using superposition
    """
    
    def __init__(
        self,
        d_model: int,
        n_features: int = 1_000_000,
        feature_dim: int = 64,
        sparsity_level: float = 0.01,
        use_efficient_computation: bool = True,
        chunk_size: int = 10000,
        feature_types: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        self.feature_dim = feature_dim
        self.sparsity_level = sparsity_level
        self.use_efficient_computation = use_efficient_computation
        self.chunk_size = chunk_size
        
        # Default feature types
        if feature_types is None:
            self.feature_types = [
                'ngram_patterns',
                'semantic_vectors', 
                'syntactic_patterns',
                'attention_patterns',
                'activation_patterns'
            ]
        else:
            self.feature_types = feature_types
        
        # Initialize feature detectors
        self._initialize_feature_detectors()
        
        # Feature importance weights (learnable)
        self.feature_importance = nn.Parameter(
            torch.ones(n_features) / math.sqrt(n_features)
        )
        
        # Sparsity mask for ablation studies
        self.register_buffer('ablation_mask', torch.ones(n_features))
        
        # Efficient computation setup
        if use_efficient_computation:
            self._setup_efficient_computation()
    
    def _initialize_feature_detectors(self):
        """Initialize different types of safety feature detectors"""
        features_per_type = self.n_features // len(self.feature_types)
        
        # N-gram pattern detectors
        self.ngram_detectors = self._create_ngram_detectors(features_per_type)
        
        # Semantic vector detectors
        self.semantic_detectors = self._create_semantic_detectors(features_per_type)
        
        # Syntactic pattern detectors  
        self.syntactic_detectors = self._create_syntactic_detectors(features_per_type)
        
        # Attention pattern detectors
        self.attention_detectors = self._create_attention_detectors(features_per_type)
        
        # Activation pattern detectors
        self.activation_detectors = self._create_activation_detectors(features_per_type)
    
    def _create_ngram_detectors(self, n_detectors: int) -> nn.ModuleDict:
        """Create n-gram pattern detectors for common harmful patterns"""
        detectors = nn.ModuleDict()
        
        # 1-gram detectors (individual tokens)
        detectors['unigram'] = nn.Linear(self.d_model, n_detectors // 4, bias=False)
        
        # 2-gram detectors (token pairs)
        detectors['bigram'] = nn.Conv1d(
            self.d_model, n_detectors // 4, kernel_size=2, padding=1
        )
        
        # 3-gram detectors (token triplets)  
        detectors['trigram'] = nn.Conv1d(
            self.d_model, n_detectors // 4, kernel_size=3, padding=1
        )
        
        # Variable-length pattern detectors
        detectors['variable'] = nn.LSTM(
            self.d_model, n_detectors // 4, batch_first=True
        )
        
        return detectors
    
    def _create_semantic_detectors(self, n_detectors: int) -> nn.ModuleDict:
        """Create semantic vector detectors for harmful concepts"""
        detectors = nn.ModuleDict()
        
        # Dense semantic embeddings
        detectors['dense'] = nn.Linear(self.d_model, n_detectors // 2, bias=False)
        
        # Sparse semantic patterns using attention
        detectors['sparse'] = nn.MultiheadAttention(
            self.d_model, num_heads=8, batch_first=True
        )
        
        # Learned semantic prototypes
        self.semantic_prototypes = nn.Parameter(
            torch.randn(n_detectors // 2, self.d_model) / math.sqrt(self.d_model)
        )
        
        return detectors
    
    def _create_syntactic_detectors(self, n_detectors: int) -> nn.ModuleDict:
        """Create syntactic pattern detectors"""
        detectors = nn.ModuleDict()
        
        # POS tag patterns
        detectors['pos_patterns'] = nn.Conv1d(
            self.d_model, n_detectors // 3, kernel_size=4, padding=2
        )
        
        # Dependency patterns
        detectors['dependency'] = nn.GRU(
            self.d_model, n_detectors // 3, batch_first=True
        )
        
        # Syntactic tree patterns
        detectors['tree_patterns'] = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=4, 
                dim_feedforward=self.d_model * 2,
                batch_first=True
            ),
            num_layers=1
        )
        
        return detectors
    
    def _create_attention_detectors(self, n_detectors: int) -> nn.ModuleDict:
        """Create attention pattern detectors"""
        detectors = nn.ModuleDict()
        
        # Self-attention patterns
        detectors['self_attention'] = nn.MultiheadAttention(
            self.d_model, num_heads=16, batch_first=True
        )
        
        # Cross-attention patterns
        detectors['cross_attention'] = nn.MultiheadAttention(
            self.d_model, num_heads=8, batch_first=True
        )
        
        return detectors
    
    def _create_activation_detectors(self, n_detectors: int) -> nn.ModuleDict:
        """Create activation pattern detectors"""
        detectors = nn.ModuleDict()
        
        # ReLU activation patterns
        detectors['relu_patterns'] = nn.Sequential(
            nn.Linear(self.d_model, n_detectors // 2),
            nn.ReLU(),
            nn.Linear(n_detectors // 2, n_detectors // 2)
        )
        
        # GELU activation patterns  
        detectors['gelu_patterns'] = nn.Sequential(
            nn.Linear(self.d_model, n_detectors // 2),
            nn.GELU(),
            nn.Linear(n_detectors // 2, n_detectors // 2)
        )
        
        return detectors
    
    def _setup_efficient_computation(self):
        """Setup for memory-efficient computation of millions of features"""
        # Create chunked computation indices
        self.chunk_indices = []
        for i in range(0, self.n_features, self.chunk_size):
            end_idx = min(i + self.chunk_size, self.n_features)
            self.chunk_indices.append((i, end_idx))
        
        # Setup gradient checkpointing
        self.use_gradient_checkpointing = True
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute activations for all safety features
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            feature_activations: [batch_size, seq_len, n_features]
        """
        if self.use_efficient_computation:
            return self._forward_efficient(hidden_states, attention_mask)
        else:
            return self._forward_standard(hidden_states, attention_mask)
    
    def _forward_standard(self, hidden_states, attention_mask):
        """Standard forward pass (for smaller models)"""
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        all_activations = []
        
        # N-gram features
        ngram_acts = self._compute_ngram_features(hidden_states, attention_mask)
        all_activations.append(ngram_acts)
        
        # Semantic features
        semantic_acts = self._compute_semantic_features(hidden_states, attention_mask)
        all_activations.append(semantic_acts)
        
        # Syntactic features
        syntactic_acts = self._compute_syntactic_features(hidden_states, attention_mask)
        all_activations.append(syntactic_acts)
        
        # Attention features
        attention_acts = self._compute_attention_features(hidden_states, attention_mask)
        all_activations.append(attention_acts)
        
        # Activation features
        activation_acts = self._compute_activation_features(hidden_states, attention_mask)
        all_activations.append(activation_acts)
        
        # Concatenate all feature activations
        feature_activations = torch.cat(all_activations, dim=-1)
        
        # Apply feature importance weights
        feature_activations = feature_activations * self.feature_importance.unsqueeze(0).unsqueeze(0)
        
        # Apply ablation mask
        feature_activations = feature_activations * self.ablation_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply sparsity
        if self.training:
            feature_activations = self._apply_sparsity(feature_activations)
        
        return feature_activations
    
    def _forward_efficient(self, hidden_states, attention_mask):
        """Memory-efficient forward pass using chunking"""
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        # Pre-allocate output tensor
        feature_activations = torch.zeros(
            batch_size, seq_len, self.n_features,
            device=device, dtype=hidden_states.dtype
        )
        
        # Process feature chunks
        for chunk_start, chunk_end in self.chunk_indices:
            chunk_size = chunk_end - chunk_start
            
            # Compute chunk features
            if self.use_gradient_checkpointing and self.training:
                chunk_acts = torch.utils.checkpoint.checkpoint(
                    self._compute_feature_chunk,
                    hidden_states, attention_mask, chunk_start, chunk_end
                )
            else:
                chunk_acts = self._compute_feature_chunk(
                    hidden_states, attention_mask, chunk_start, chunk_end
                )
            
            # Store in output tensor
            feature_activations[:, :, chunk_start:chunk_end] = chunk_acts
        
        # Apply global operations
        feature_activations = feature_activations * self.feature_importance.unsqueeze(0).unsqueeze(0)
        feature_activations = feature_activations * self.ablation_mask.unsqueeze(0).unsqueeze(0)
        
        if self.training:
            feature_activations = self._apply_sparsity(feature_activations)
            
        return feature_activations
    
    def _compute_feature_chunk(self, hidden_states, attention_mask, start_idx, end_idx):
        """Compute a chunk of features efficiently"""
        chunk_size = end_idx - start_idx
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Determine which feature type this chunk belongs to
        features_per_type = self.n_features // len(self.feature_types)
        type_idx = start_idx // features_per_type
        within_type_start = start_idx % features_per_type
        within_type_end = within_type_start + chunk_size
        
        if type_idx == 0:  # N-gram features
            return self._compute_ngram_features(hidden_states, attention_mask)[
                :, :, within_type_start:within_type_end
            ]
        elif type_idx == 1:  # Semantic features
            return self._compute_semantic_features(hidden_states, attention_mask)[
                :, :, within_type_start:within_type_end
            ]
        elif type_idx == 2:  # Syntactic features
            return self._compute_syntactic_features(hidden_states, attention_mask)[
                :, :, within_type_start:within_type_end
            ]
        elif type_idx == 3:  # Attention features
            return self._compute_attention_features(hidden_states, attention_mask)[
                :, :, within_type_start:within_type_end
            ]
        else:  # Activation features
            return self._compute_activation_features(hidden_states, attention_mask)[
                :, :, within_type_start:within_type_end
            ]
    
    def _compute_ngram_features(self, hidden_states, attention_mask):
        """Compute n-gram based safety features"""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Unigram features
        unigram_acts = self.ngram_detectors['unigram'](hidden_states)
        
        # Bigram features
        bigram_input = rearrange(hidden_states, 'b s d -> b d s')
        bigram_acts = self.ngram_detectors['bigram'](bigram_input)
        bigram_acts = rearrange(bigram_acts, 'b d s -> b s d')
        # Ensure same sequence length
        if bigram_acts.size(1) != seq_len:
            bigram_acts = bigram_acts[:, :seq_len, :]
        
        # Trigram features
        trigram_acts = self.ngram_detectors['trigram'](bigram_input)
        trigram_acts = rearrange(trigram_acts, 'b d s -> b s d')
        # Ensure same sequence length
        if trigram_acts.size(1) != seq_len:
            trigram_acts = trigram_acts[:, :seq_len, :]
        
        # Variable length features
        variable_acts, _ = self.ngram_detectors['variable'](hidden_states)
        
        return torch.cat([unigram_acts, bigram_acts, trigram_acts, variable_acts], dim=-1)
    
    def _compute_semantic_features(self, hidden_states, attention_mask):
        """Compute semantic safety features"""
        # Dense semantic features
        dense_acts = self.semantic_detectors['dense'](hidden_states)
        
        # Sparse attention-based features
        sparse_acts, _ = self.semantic_detectors['sparse'](
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # Prototype-based features
        prototype_similarities = torch.matmul(
            hidden_states, self.semantic_prototypes.T
        )
        
        return torch.cat([dense_acts, prototype_similarities], dim=-1)
    
    def _compute_syntactic_features(self, hidden_states, attention_mask):
        """Compute syntactic safety features"""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # POS pattern features
        pos_input = rearrange(hidden_states, 'b s d -> b d s')
        pos_acts = self.syntactic_detectors['pos_patterns'](pos_input)
        pos_acts = rearrange(pos_acts, 'b d s -> b s d')
        # Ensure same sequence length
        if pos_acts.size(1) != seq_len:
            pos_acts = pos_acts[:, :seq_len, :]
        
        # Dependency features
        dep_acts, _ = self.syntactic_detectors['dependency'](hidden_states)
        
        # Tree pattern features (simplified)
        tree_acts = self.syntactic_detectors['tree_patterns'](hidden_states)
        
        return torch.cat([pos_acts, dep_acts, tree_acts], dim=-1)
    
    def _compute_attention_features(self, hidden_states, attention_mask):
        """Compute attention pattern safety features"""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Self-attention features
        self_att_acts, _ = self.attention_detectors['self_attention'](
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # Cross-attention features (using learned queries from semantic prototypes)
        # Use a subset of semantic prototypes as queries
        n_queries = min(seq_len, self.semantic_prototypes.size(0))
        learned_queries = self.semantic_prototypes[:n_queries].unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        cross_att_acts, _ = self.attention_detectors['cross_attention'](
            learned_queries, hidden_states, hidden_states,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # If cross attention queries are fewer than seq_len, repeat or pad
        if cross_att_acts.size(1) < seq_len:
            repeat_factor = seq_len // cross_att_acts.size(1) + 1
            cross_att_acts = cross_att_acts.repeat(1, repeat_factor, 1)
        cross_att_acts = cross_att_acts[:, :seq_len, :]
        
        return torch.cat([self_att_acts, cross_att_acts], dim=-1)
    
    def _compute_activation_features(self, hidden_states, attention_mask):
        """Compute activation pattern safety features"""
        # ReLU activation features
        relu_acts = self.activation_detectors['relu_patterns'](hidden_states)
        
        # GELU activation features
        gelu_acts = self.activation_detectors['gelu_patterns'](hidden_states)
        
        return torch.cat([relu_acts, gelu_acts], dim=-1)
    
    def _apply_sparsity(self, feature_activations):
        """Apply sparsity to feature activations during training"""
        if not self.training:
            return feature_activations
        
        # Top-k sparsity
        k = int(self.n_features * self.sparsity_level)
        values, indices = torch.topk(feature_activations.abs(), k, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(feature_activations)
        sparse_mask.scatter_(-1, indices, 1.0)
        
        return feature_activations * sparse_mask
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get current feature importance scores"""
        return self.feature_importance.data
    
    def ablate_features(self, feature_indices: torch.Tensor):
        """Ablate specific features by setting their mask to 0"""
        self.ablation_mask[feature_indices] = 0.0
    
    def restore_features(self, feature_indices: torch.Tensor):
        """Restore previously ablated features"""
        self.ablation_mask[feature_indices] = 1.0
    
    def get_active_features(self, hidden_states, attention_mask=None, threshold=0.1):
        """Get indices of currently active features"""
        with torch.no_grad():
            activations = self.forward(hidden_states, attention_mask)
            max_activations = activations.max(dim=1)[0].max(dim=0)[0]  # Max over batch and sequence
            active_indices = torch.where(max_activations > threshold)[0]
            return active_indices 