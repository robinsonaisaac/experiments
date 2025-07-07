import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model, GPT2Block
from transformers import GPT2Config, AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from .mfs_layer import MFSLayer, EfficientMFSLayer
from .aggregators import ErrorCorrectingAggregator, SimpleAggregator


class MFSTransformerConfig:
    """Configuration class for MFS-enhanced transformer"""
    
    def __init__(
        self,
        base_model_name: str = "gpt2",
        d_model: int = 768,
        n_layers: int = 12,
        n_safety_features: int = 1_000_000,
        feature_dim: int = 64,
        mfs_layers: Optional[list] = None,  # Which layers to add MFS to
        sparsity_level: float = 0.01,
        aggregation_type: str = "error_correcting",
        use_efficient_computation: bool = True,
        chunk_size: int = 10000,
        safety_alpha: float = 1.0,  # Weight for safety loss
        capability_preservation: bool = True,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_safety_features = n_safety_features
        self.feature_dim = feature_dim
        
        # Default: add MFS to every 4th layer
        if mfs_layers is None:
            self.mfs_layers = list(range(0, n_layers, 4))
        else:
            self.mfs_layers = mfs_layers
            
        self.sparsity_level = sparsity_level
        self.aggregation_type = aggregation_type
        self.use_efficient_computation = use_efficient_computation
        self.chunk_size = chunk_size
        self.safety_alpha = safety_alpha
        self.capability_preservation = capability_preservation
        
        # Additional config parameters
        for key, value in kwargs.items():
            setattr(self, key, value)


class MFSTransformer(PreTrainedModel):
    """
    MFS-enhanced transformer model for robust safety
    """
    
    def __init__(self, config: MFSTransformerConfig):
        # Initialize as PreTrainedModel for compatibility
        super().__init__(AutoConfig.from_pretrained(config.base_model_name))
        
        self.config = config
        self.safety_alpha = config.safety_alpha
        
        # Load base transformer
        self.base_model = AutoModel.from_pretrained(config.base_model_name)
        self.base_config = self.base_model.config
        
        # Get actual model dimensions
        self.d_model = getattr(self.base_config, 'hidden_size', config.d_model)
        self.n_layers = getattr(self.base_config, 'num_hidden_layers', config.n_layers)
        
        # Initialize MFS layers
        self.mfs_layers = nn.ModuleDict()
        for layer_idx in config.mfs_layers:
            if layer_idx < self.n_layers:
                if config.use_efficient_computation and config.n_safety_features > 100_000:
                    mfs_layer = EfficientMFSLayer(
                        d_model=self.d_model,
                        n_safety_features=config.n_safety_features,
                        feature_dim=config.feature_dim,
                        sparsity_level=config.sparsity_level,
                        layer_idx=layer_idx,
                        chunk_size=config.chunk_size
                    )
                else:
                    mfs_layer = MFSLayer(
                        d_model=self.d_model,
                        n_safety_features=config.n_safety_features,
                        feature_dim=config.feature_dim,
                        sparsity_level=config.sparsity_level,
                        layer_idx=layer_idx,
                        use_efficient_computation=config.use_efficient_computation,
                        chunk_size=config.chunk_size
                    )
                
                self.mfs_layers[str(layer_idx)] = mfs_layer
        
        # Safety head for final safety scoring
        self.safety_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Task head (if we want to maintain capabilities)
        if hasattr(self.base_model, 'lm_head'):
            self.lm_head = self.base_model.lm_head
        else:
            # Create a simple language modeling head
            vocab_size = getattr(self.base_config, 'vocab_size', 50257)
            self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        
        # Feature importance tracking
        self.register_buffer('global_feature_importance', 
                           torch.zeros(config.n_safety_features))
        self.register_buffer('safety_activations_history',
                           torch.zeros(100))  # Track last 100 safety scores
        self.history_idx = 0
        
        # Computational cost tracking
        self.flops_per_forward = 0
        self.memory_usage_mb = 0
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        safety_labels: Optional[torch.Tensor] = None,
        return_safety_scores: bool = True,
        return_feature_importance: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through MFS-enhanced transformer
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] for language modeling
            safety_labels: [batch_size] for safety classification
            return_safety_scores: Whether to compute safety scores
            return_feature_importance: Whether to return feature importance
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Dictionary containing loss, logits, safety_scores, etc.
        """
        batch_size, seq_len = input_ids.shape
        
        # Forward through base model with hooks for MFS layers
        if hasattr(self.base_model, 'transformer'):
            # GPT-style model
            embeddings = self.base_model.transformer.wte(input_ids)
            if hasattr(self.base_model.transformer, 'wpe'):
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                embeddings += self.base_model.transformer.wpe(position_ids)
            
            hidden_states = embeddings
            all_hidden_states = [] if output_hidden_states else None
            all_safety_scores = []
            
            # Forward through transformer layers with MFS integration
            for layer_idx, layer in enumerate(self.base_model.transformer.h):
                # Apply base transformer layer
                if attention_mask is not None:
                    # Convert attention mask for transformer
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                else:
                    extended_attention_mask = None
                
                # Forward through base layer
                layer_outputs = layer(hidden_states, attention_mask=extended_attention_mask)
                hidden_states = layer_outputs[0]
                
                # Apply MFS layer if present
                if str(layer_idx) in self.mfs_layers:
                    mfs_layer = self.mfs_layers[str(layer_idx)]
                    hidden_states, safety_scores = mfs_layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        return_safety_scores=return_safety_scores
                    )
                    
                    if safety_scores is not None:
                        all_safety_scores.append(safety_scores)
                
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
            
            # Apply final layer norm
            if hasattr(self.base_model.transformer, 'ln_f'):
                hidden_states = self.base_model.transformer.ln_f(hidden_states)
                
        else:
            # BERT-style or other model
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            all_hidden_states = base_outputs.hidden_states
            all_safety_scores = []
            
            # Apply MFS layers to specified hidden states
            for layer_idx in self.mfs_layers:
                if layer_idx < len(all_hidden_states):
                    mfs_layer = self.mfs_layers[str(layer_idx)]
                    modified_states, safety_scores = mfs_layer(
                        all_hidden_states[layer_idx],
                        attention_mask=attention_mask,
                        return_safety_scores=return_safety_scores
                    )
                    
                    # Update the hidden states
                    all_hidden_states = list(all_hidden_states)
                    all_hidden_states[layer_idx] = modified_states
                    
                    if safety_scores is not None:
                        all_safety_scores.append(safety_scores)
            
            hidden_states = all_hidden_states[-1]
        
        # Compute final safety scores
        final_safety_scores = None
        if return_safety_scores:
            if all_safety_scores:
                # Average safety scores across MFS layers
                stacked_scores = torch.stack(all_safety_scores, dim=0)
                avg_safety_scores = stacked_scores.mean(dim=0)
            else:
                # Compute from final hidden states
                pooled_hidden = hidden_states.mean(dim=1)  # Pool sequence length
                avg_safety_scores = self.safety_head(pooled_hidden).squeeze(-1)
            
            final_safety_scores = avg_safety_scores
            
            # Update safety history
            if self.training:
                self._update_safety_history(final_safety_scores)
        
        # Compute language modeling logits
        lm_logits = self.lm_head(hidden_states)
        
        # Compute losses
        total_loss = None
        lm_loss = None
        safety_loss = None
        
        if labels is not None or safety_labels is not None:
            total_loss = 0.0
            
            # Language modeling loss
            if labels is not None:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                  shift_labels.view(-1))
                total_loss += lm_loss
            
            # Safety loss
            if safety_labels is not None and final_safety_scores is not None:
                safety_loss_fct = nn.BCELoss()
                safety_loss = safety_loss_fct(final_safety_scores, safety_labels.float())
                total_loss += self.safety_alpha * safety_loss
        
        # Compute feature importance if requested
        feature_importance = None
        if return_feature_importance:
            feature_importance = self._compute_feature_importance()
        
        # Prepare output
        output = {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'safety_loss': safety_loss,
            'logits': lm_logits,
            'safety_scores': final_safety_scores,
            'hidden_states': all_hidden_states if output_hidden_states else None,
            'feature_importance': feature_importance,
        }
        
        return output
    
    def _update_safety_history(self, safety_scores: torch.Tensor):
        """Update running history of safety scores"""
        avg_score = safety_scores.mean().item()
        self.safety_activations_history[self.history_idx] = avg_score
        self.history_idx = (self.history_idx + 1) % self.safety_activations_history.size(0)
    
    def _compute_feature_importance(self) -> torch.Tensor:
        """Compute global feature importance across all MFS layers"""
        total_importance = torch.zeros_like(self.global_feature_importance)
        
        for mfs_layer in self.mfs_layers.values():
            layer_importance = mfs_layer.get_feature_importance()
            total_importance += layer_importance
        
        # Normalize
        if total_importance.sum() > 0:
            total_importance = total_importance / total_importance.sum()
        
        return total_importance
    
    def ablate_features(self, feature_indices: torch.Tensor, layers: Optional[list] = None):
        """Ablate specific features across specified layers"""
        if layers is None:
            layers = list(self.mfs_layers.keys())
        
        for layer_key in layers:
            if layer_key in self.mfs_layers:
                self.mfs_layers[layer_key].ablate_features(feature_indices)
    
    def get_computational_cost(self) -> Dict[str, Any]:
        """Get computational cost breakdown"""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        mfs_params = sum(p.numel() for p in self.mfs_layers.parameters())
        safety_params = sum(p.numel() for p in self.safety_head.parameters())
        
        total_memory = 0
        total_flops = 0
        
        for mfs_layer in self.mfs_layers.values():
            cost_info = mfs_layer.get_computational_cost()
            total_memory += cost_info['memory_mb']
            total_flops += cost_info['flops_per_token']
        
        return {
            'base_model_params': base_params,
            'mfs_params': mfs_params,
            'safety_params': safety_params,
            'total_params': base_params + mfs_params + safety_params,
            'memory_mb': total_memory,
            'flops_per_token': total_flops,
            'mfs_layers_count': len(self.mfs_layers),
            'features_per_layer': self.config.n_safety_features,
        }
    
    def generate_safe(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        safety_threshold: float = 0.5,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with safety filtering
        
        Args:
            input_ids: [batch_size, seq_len]
            max_length: Maximum generation length
            safety_threshold: Threshold for safety filtering
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Dictionary with generated text and safety information
        """
        self.eval()
        generated_ids = input_ids.clone()
        safety_scores_history = []
        
        with torch.no_grad():
            for step in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(
                    generated_ids,
                    return_safety_scores=True
                )
                
                logits = outputs['logits'][:, -1, :]  # Last token logits
                safety_scores = outputs['safety_scores']
                
                safety_scores_history.append(safety_scores.cpu())
                
                # Check safety
                if safety_scores.max() > safety_threshold:
                    # Safety violation detected - apply intervention
                    # Option 1: Stop generation
                    break
                    
                    # Option 2: Sample from safer subset of vocabulary
                    # safe_vocab_mask = self._get_safe_vocab_mask(logits)
                    # logits = logits.masked_fill(~safe_vocab_mask, -float('inf'))
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                if generation_kwargs.get('do_sample', True):
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = probs.argmax(dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for EOS token
                if next_token.item() == self.base_model.config.eos_token_id:
                    break
        
        return {
            'generated_ids': generated_ids,
            'safety_scores_history': torch.stack(safety_scores_history) if safety_scores_history else None,
            'generation_stopped_by_safety': len(safety_scores_history) < max_length - input_ids.size(1)
        }
    
    def measure_attack_resistance(self, attack_budget: int = 1000) -> Dict[str, float]:
        """
        Measure resistance to feature ablation attacks
        
        Args:
            attack_budget: Number of features attacker can ablate
            
        Returns:
            Dictionary with resistance metrics
        """
        original_importance = self._compute_feature_importance()
        
        # Simulate greedy attack - ablate most important features
        top_features = torch.topk(original_importance, attack_budget)[1]
        
        # Store original state
        original_masks = {}
        for layer_key, mfs_layer in self.mfs_layers.items():
            original_masks[layer_key] = mfs_layer.safety_features.ablation_mask.clone()
        
        # Apply attack
        self.ablate_features(top_features)
        
        # Measure degradation (simplified metric)
        post_attack_importance = self._compute_feature_importance()
        importance_degradation = 1.0 - (post_attack_importance.sum() / original_importance.sum()).item()
        
        # Restore original state
        for layer_key, mfs_layer in self.mfs_layers.items():
            mfs_layer.safety_features.ablation_mask.copy_(original_masks[layer_key])
        
        return {
            'importance_degradation': importance_degradation,
            'features_attacked': attack_budget,
            'total_features': self.config.n_safety_features,
            'attack_fraction': attack_budget / self.config.n_safety_features,
            'resistance_score': 1.0 - importance_degradation  # Higher is better
        }
    
    def get_safety_statistics(self) -> Dict[str, float]:
        """Get current safety statistics"""
        history = self.safety_activations_history
        valid_history = history[history > 0]  # Filter out zeros
        
        if len(valid_history) == 0:
            return {'mean_safety': 0.0, 'std_safety': 0.0, 'max_safety': 0.0}
        
        return {
            'mean_safety': valid_history.mean().item(),
            'std_safety': valid_history.std().item(),
            'max_safety': valid_history.max().item(),
            'min_safety': valid_history.min().item(),
        }


class MFSTransformerBlock(nn.Module):
    """
    Transformer block enhanced with MFS safety layer
    """
    
    def __init__(
        self,
        config: GPT2Config,
        layer_idx: int = 0,
        mfs_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Standard transformer components
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Block(config, layer_idx=layer_idx).attn  # Reuse GPT2 attention
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2Block(config, layer_idx=layer_idx).mlp   # Reuse GPT2 MLP
        
        # MFS safety layer
        if mfs_config is not None:
            self.mfs_layer = self._create_mfs_layer(config.n_embd, mfs_config)
            self.use_mfs = True
        else:
            self.mfs_layer = None
            self.use_mfs = False
        
        # Integration method
        self.mfs_integration = mfs_config.get('integration_method', 'residual') if mfs_config else 'residual'
        
    def _create_mfs_layer(self, d_model: int, mfs_config: Dict[str, Any]) -> MFSLayer:
        """Create MFS layer based on configuration"""
        if mfs_config.get('use_efficient', True):
            return EfficientMFSLayer(
                d_model=d_model,
                n_safety_features=mfs_config.get('n_safety_features', 100000),
                feature_dim=mfs_config.get('feature_dim', 64),
                sparsity_level=mfs_config.get('sparsity_level', 0.01),
                layer_idx=self.layer_idx,
                chunk_size=mfs_config.get('chunk_size', 10000),
            )
        else:
            return MFSLayer(
                d_model=d_model,
                n_safety_features=mfs_config.get('n_safety_features', 100000),
                feature_dim=mfs_config.get('feature_dim', 64),
                sparsity_level=mfs_config.get('sparsity_level', 0.01),
                layer_idx=self.layer_idx,
                use_efficient_computation=False,
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        return_safety_scores: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Self-attention
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: [batch_size, seq_len, n_embd]
        outputs = attn_outputs[1:]
        
        # Apply residual connection
        hidden_states = attn_output + residual
        
        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        
        # Apply MFS layer if enabled
        safety_scores = None
        if self.use_mfs and self.mfs_layer is not None:
            if self.mfs_integration == 'residual':
                # Residual connection with MFS
                mfs_output, safety_scores = self.mfs_layer(
                    hidden_states, 
                    attention_mask=attention_mask,
                    return_safety_scores=return_safety_scores
                )
                hidden_states = hidden_states + mfs_output
            elif self.mfs_integration == 'replace':
                # Replace hidden states with MFS output
                hidden_states, safety_scores = self.mfs_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_safety_scores=return_safety_scores
                )
            elif self.mfs_integration == 'parallel':
                # Parallel processing and averaging
                mfs_output, safety_scores = self.mfs_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_safety_scores=return_safety_scores
                )
                hidden_states = (hidden_states + mfs_output) / 2.0
        
        # Prepare outputs
        if return_safety_scores and safety_scores is not None:
            outputs = (hidden_states, safety_scores) + outputs
        else:
            outputs = (hidden_states,) + outputs
        
        return outputs


class MFSTransformer(GPT2Model):
    """
    GPT2 model enhanced with Massive Feature Superposition for safety
    """
    
    def __init__(self, config: GPT2Config, mfs_config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.mfs_config = mfs_config or {}
        
        # Replace transformer blocks with MFS-enhanced blocks
        self.h = nn.ModuleList([
            MFSTransformerBlock(config, layer_idx=i, mfs_config=self._get_layer_mfs_config(i))
            for i in range(config.n_layer)
        ])
        
        # Safety aggregation across layers
        if self.mfs_config.get('use_global_aggregation', True):
            self.global_safety_aggregator = nn.Sequential(
                nn.Linear(config.n_layer, config.n_layer // 2),
                nn.GELU(),
                nn.Linear(config.n_layer // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.global_safety_aggregator = None
        
        # Initialize weights
        self.init_weights()
    
    def _get_layer_mfs_config(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        """Get MFS configuration for a specific layer"""
        if not self.mfs_config.get('enabled', True):
            return None
        
        # Apply MFS to specific layers only
        mfs_layers = self.mfs_config.get('mfs_layers', 'all')
        if mfs_layers == 'all':
            return self.mfs_config
        elif isinstance(mfs_layers, list) and layer_idx in mfs_layers:
            return self.mfs_config
        elif isinstance(mfs_layers, str) and mfs_layers == 'middle':
            # Apply to middle layers
            total_layers = self.config.n_layer
            start_layer = total_layers // 4
            end_layer = 3 * total_layers // 4
            if start_layer <= layer_idx <= end_layer:
                return self.mfs_config
        
        return None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_safety_scores: bool = False,
    ):
        
        # Standard GPT2 forward pass setup
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        
        # Prepare past key values
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # Attention mask
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Head mask
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        
        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        
        hidden_states = self.drop(hidden_states)
        
        output_shape = input_shape + (hidden_states.size(-1),)
        
        # Transformer blocks
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_safety_scores = [] if return_safety_scores else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_safety_scores=return_safety_scores,
            )
            
            # Extract outputs
            if return_safety_scores and len(outputs) > 1 and outputs[1] is not None:
                hidden_states = outputs[0]
                layer_safety_scores = outputs[1]
                all_safety_scores.append(layer_safety_scores)
                remaining_outputs = outputs[2:]
            else:
                hidden_states = outputs[0]
                remaining_outputs = outputs[1:]
            
            if use_cache:
                presents = presents + (remaining_outputs[0],)
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (remaining_outputs[1 if use_cache else 0],)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Aggregate safety scores across layers
        global_safety_score = None
        if return_safety_scores and all_safety_scores:
            if self.global_safety_aggregator is not None:
                # Stack safety scores from MFS layers: [batch_size, n_mfs_layers]
                stacked_safety_scores = torch.stack(all_safety_scores, dim=1)
                
                # Check if aggregator input size matches
                n_mfs_layers = len(all_safety_scores)
                expected_input_size = self.global_safety_aggregator[0].in_features
                
                if n_mfs_layers != expected_input_size:
                    # Create a new aggregator with correct input size
                    self.global_safety_aggregator = nn.Sequential(
                        nn.Linear(n_mfs_layers, max(n_mfs_layers // 2, 1)),
                        nn.GELU(),
                        nn.Linear(max(n_mfs_layers // 2, 1), 1),
                        nn.Sigmoid()
                    ).to(stacked_safety_scores.device)
                
                # Aggregate to single score per sample: [batch_size, 1]
                global_safety_score = self.global_safety_aggregator(stacked_safety_scores).squeeze(-1)
            else:
                # Simple averaging if no global aggregator
                global_safety_score = torch.stack(all_safety_scores, dim=1).mean(dim=1)
        
        # Prepare final outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
        }
        
        if return_safety_scores:
            outputs['safety_scores'] = global_safety_score
            outputs['layer_safety_scores'] = all_safety_scores
        
        if not return_dict:
            return tuple(v for v in outputs.values() if v is not None)
        
        return outputs


class MFSForCausalLM(GPT2LMHeadModel):
    """
    MFS-enhanced GPT2 for causal language modeling
    """
    
    def __init__(self, config: GPT2Config, mfs_config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Replace transformer with MFS-enhanced version
        self.transformer = MFSTransformer(config, mfs_config)
        
        # Safety head for explicit safety scoring
        if mfs_config and mfs_config.get('use_safety_head', True):
            self.safety_head = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.n_embd // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.safety_head = None
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_safety_scores: bool = False,
        safety_labels: Optional[torch.FloatTensor] = None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_safety_scores=return_safety_scores,
        )
        
        hidden_states = transformer_outputs['last_hidden_state'] if return_dict else transformer_outputs[0]
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Safety head
        safety_logits = None
        if self.safety_head is not None:
            # Pool hidden states for safety prediction
            pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, n_embd]
            safety_logits = self.safety_head(pooled_hidden).squeeze(-1)  # [batch_size]
        
        # Compute losses
        lm_loss = None
        safety_loss = None
        total_loss = None
        
        if labels is not None:
            # Language modeling loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss = lm_loss
        
        if safety_labels is not None and safety_logits is not None:
            # Safety classification loss
            safety_loss_fct = nn.BCELoss()
            safety_loss = safety_loss_fct(safety_logits, safety_labels.float())
            
            if total_loss is not None:
                # Combine losses with weighting
                safety_weight = 0.1  # Adjustable weight
                total_loss = lm_loss + safety_weight * safety_loss
            else:
                total_loss = safety_loss
        
        # Prepare outputs
        outputs = {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'safety_loss': safety_loss,
            'logits': lm_logits,
            'safety_logits': safety_logits,
            'past_key_values': transformer_outputs.get('past_key_values'),
            'hidden_states': transformer_outputs.get('hidden_states'),
            'attentions': transformer_outputs.get('attentions'),
        }
        
        if return_safety_scores:
            outputs['safety_scores'] = transformer_outputs.get('safety_scores')
            outputs['layer_safety_scores'] = transformer_outputs.get('layer_safety_scores')
        
        if not return_dict:
            return tuple(v for v in outputs.values() if v is not None)
        
        return outputs
    
    def get_safety_analysis(self, input_ids, attention_mask=None):
        """
        Get detailed safety analysis for input
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_safety_scores=True,
                output_hidden_states=True,
            )
            
            analysis = {
                'global_safety_score': outputs['safety_scores'],
                'layer_safety_scores': outputs['layer_safety_scores'],
                'safety_logits': outputs['safety_logits'],
                'is_safe': outputs['safety_logits'] > 0.5 if outputs['safety_logits'] is not None else None,
            }
            
            # Get feature importance from MFS layers
            feature_importance = {}
            for i, block in enumerate(self.transformer.h):
                if hasattr(block, 'mfs_layer') and block.mfs_layer is not None:
                    importance = block.mfs_layer.get_feature_importance()
                    feature_importance[f'layer_{i}'] = importance
            
            analysis['feature_importance'] = feature_importance
            
            return analysis
    
    def ablate_safety_features(self, layer_idx: int, feature_indices: torch.Tensor):
        """Ablate specific safety features for robustness testing"""
        if layer_idx < len(self.transformer.h):
            block = self.transformer.h[layer_idx]
            if hasattr(block, 'mfs_layer') and block.mfs_layer is not None:
                block.mfs_layer.ablate_features(feature_indices)


def create_mfs_model(
    model_size: str = "gpt2",
    n_safety_features: int = 100000,
    mfs_layers: str = "middle",
    use_efficient: bool = True,
    **kwargs
) -> MFSForCausalLM:
    """
    Factory function to create MFS models of different sizes
    
    Args:
        model_size: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", or custom config
        n_safety_features: Number of safety features per MFS layer
        mfs_layers: Which layers to apply MFS to ("all", "middle", or list of indices)
        use_efficient: Whether to use efficient MFS implementation
    """
    
    # Load base configuration
    if isinstance(model_size, str):
        config = GPT2Config.from_pretrained(model_size)
    else:
        config = model_size  # Assume it's already a config
    
    # MFS configuration
    mfs_config = {
        'enabled': True,
        'n_safety_features': n_safety_features,
        'feature_dim': 64,
        'sparsity_level': 0.01,
        'mfs_layers': mfs_layers,
        'use_efficient': use_efficient,
        'chunk_size': 10000,
        'integration_method': 'residual',
        'use_global_aggregation': True,
        'use_safety_head': True,
        **kwargs
    }
    
    # Create model
    model = MFSForCausalLM(config, mfs_config)
    
    return model 