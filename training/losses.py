import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class MFSLoss(nn.Module):
    """
    Multi-objective loss for MFS training combining:
    1. Task loss (language modeling)
    2. Safety loss (safety classification)
    3. Diversity loss (feature diversity)
    4. Robustness loss (feature redundancy)
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        safety_weight: float = 0.5,
        diversity_weight: float = 0.1,
        robustness_weight: float = 0.05,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.task_weight = task_weight
        self.safety_weight = safety_weight  
        self.diversity_weight = diversity_weight
        self.robustness_weight = robustness_weight
        self.temperature = temperature
        
        # Individual loss components
        self.task_loss = TaskLoss()
        self.safety_loss = SafetyLoss()
        self.diversity_loss = DiversityLoss()
        self.robustness_loss = RobustnesslLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.LongTensor,
        safety_labels: Optional[torch.FloatTensor] = None,
        feature_activations: Optional[List[torch.Tensor]] = None,
        return_individual_losses: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-objective loss
        
        Args:
            outputs: Model outputs containing logits, safety_logits, etc.
            labels: Language modeling labels
            safety_labels: Safety classification labels
            feature_activations: List of feature activations from MFS layers
            return_individual_losses: Whether to return breakdown of losses
        """
        
        losses = {}
        
        # Task loss (language modeling)
        if 'logits' in outputs:
            losses['task'] = self.task_loss(outputs['logits'], labels)
        else:
            losses['task'] = torch.tensor(0.0, device=labels.device)
        
        # Safety loss
        if 'safety_logits' in outputs and safety_labels is not None:
            losses['safety'] = self.safety_loss(outputs['safety_logits'], safety_labels)
        else:
            losses['safety'] = torch.tensor(0.0, device=labels.device)
        
        # Diversity loss (encourages diverse feature usage)
        if feature_activations is not None:
            losses['diversity'] = self.diversity_loss(feature_activations)
        else:
            losses['diversity'] = torch.tensor(0.0, device=labels.device)
        
        # Robustness loss (encourages feature redundancy)
        if feature_activations is not None:
            losses['robustness'] = self.robustness_loss(feature_activations)
        else:
            losses['robustness'] = torch.tensor(0.0, device=labels.device)
        
        # Combine losses
        total_loss = (
            self.task_weight * losses['task'] +
            self.safety_weight * losses['safety'] +
            self.diversity_weight * losses['diversity'] +
            self.robustness_weight * losses['robustness']
        )
        
        if return_individual_losses:
            return total_loss, losses
        else:
            return total_loss


class TaskLoss(nn.Module):
    """Standard language modeling loss"""
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = self.loss_fct(shift_logits, shift_labels)
        return loss


class SafetyLoss(nn.Module):
    """
    Safety-focused loss function
    """
    
    def __init__(
        self,
        safety_weight: float = 1.0,
        margin: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.safety_weight = safety_weight
        self.margin = margin
        self.temperature = temperature
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        safety_scores: torch.Tensor,
        safety_labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute safety loss
        
        Args:
            safety_scores: [batch_size] predicted safety scores (0-1)
            safety_labels: [batch_size] true safety labels (0=safe, 1=unsafe)
        """
        # Basic BCE loss
        bce_loss = self.bce_loss(safety_scores, safety_labels.float())
        
        # Margin-based loss to encourage stronger separation
        safe_mask = safety_labels == 0
        unsafe_mask = safety_labels == 1
        
        margin_loss = 0.0
        if safe_mask.any():
            # Safe examples should have low scores (< margin)
            safe_scores = safety_scores[safe_mask]
            margin_loss += F.relu(safe_scores - self.margin).mean()
        
        if unsafe_mask.any():
            # Unsafe examples should have high scores (> 1 - margin)
            unsafe_scores = safety_scores[unsafe_mask]
            margin_loss += F.relu((1.0 - self.margin) - unsafe_scores).mean()
        
        total_loss = bce_loss + 0.1 * margin_loss
        return self.safety_weight * total_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss to encourage feature diversity and prevent collapse
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.1,
        correlation_penalty: float = 0.01,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.correlation_penalty = correlation_penalty
        self.entropy_weight = entropy_weight
    
    def forward(
        self,
        feature_activations: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute diversity loss
        
        Args:
            feature_activations: [batch_size, seq_len, n_features] feature activations
        """
        batch_size, seq_len, n_features = feature_activations.shape
        
        # Flatten features for correlation computation
        flat_features = feature_activations.view(-1, n_features)  # [batch_size * seq_len, n_features]
        
        # 1. Correlation penalty - penalize highly correlated features
        correlation_loss = 0.0
        if n_features > 1:
            # Compute correlation matrix (use subset if too large)
            max_features_for_corr = min(n_features, 1000)
            if n_features > max_features_for_corr:
                # Sample random subset
                indices = torch.randperm(n_features)[:max_features_for_corr]
                subset_features = flat_features[:, indices]
            else:
                subset_features = flat_features
            
            # Compute pairwise correlations
            normalized_features = F.normalize(subset_features, dim=0)
            correlation_matrix = torch.mm(normalized_features.T, normalized_features) / subset_features.size(0)
            
            # Penalize off-diagonal correlations
            mask = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device)
            off_diagonal_corr = correlation_matrix * (1 - mask)
            correlation_loss = off_diagonal_corr.abs().mean()
        
        # 2. Entropy-based diversity - encourage uniform activation distribution
        entropy_loss = 0.0
        if flat_features.size(0) > 1:
            # Compute activation probabilities
            activation_probs = torch.sigmoid(flat_features)  # [batch_size * seq_len, n_features]
            
            # Mean activation per feature
            mean_activations = activation_probs.mean(dim=0)  # [n_features]
            
            # Entropy of each feature's activation distribution
            epsilon = 1e-8
            entropy = -(mean_activations * torch.log(mean_activations + epsilon) + 
                       (1 - mean_activations) * torch.log(1 - mean_activations + epsilon))
            
            # We want high entropy (uniform distribution)
            entropy_loss = -entropy.mean()
        
        # 3. Feature utilization - penalize unused features
        utilization_loss = 0.0
        feature_usage = (flat_features.abs() > 0.01).float().mean(dim=0)  # [n_features]
        min_usage_threshold = 0.001  # At least 0.1% usage
        utilization_loss = F.relu(min_usage_threshold - feature_usage).mean()
        
        total_diversity_loss = (
            self.correlation_penalty * correlation_loss + 
            self.entropy_weight * entropy_loss +
            0.01 * utilization_loss
        )
        
        return self.diversity_weight * total_diversity_loss


class RobustnesslLoss(nn.Module):
    """
    Encourages feature redundancy for robustness against attacks
    Features should have overlapping functionality
    """
    
    def __init__(self, similarity_threshold: float = 0.7, margin: float = 0.1):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.margin = margin
    
    def forward(self, feature_activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Encourage feature redundancy by maximizing similarity between 
        related features while maintaining some diversity
        """
        total_robustness_loss = 0.0
        
        for activations in feature_activations:
            batch_size, seq_len, n_features = activations.shape
            
            # Sample a subset of features for efficiency
            max_features_to_compare = min(n_features, 1000)
            if n_features > max_features_to_compare:
                feature_indices = torch.randperm(n_features)[:max_features_to_compare]
                sampled_activations = activations[:, :, feature_indices]
            else:
                sampled_activations = activations
            
            # Compute pairwise similarities
            layer_loss = self._compute_redundancy_loss(sampled_activations)
            total_robustness_loss += layer_loss
        
        return total_robustness_loss / len(feature_activations)
    
    def _compute_redundancy_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute redundancy loss for a layer's activations"""
        batch_size, seq_len, n_features = activations.shape
        
        # Flatten across batch and sequence for correlation computation
        flat_activations = activations.view(-1, n_features)  # [batch_size * seq_len, n_features]
        
        # Compute correlation matrix between features
        centered_activations = flat_activations - flat_activations.mean(dim=0, keepdim=True)
        correlation_matrix = torch.mm(centered_activations.T, centered_activations)
        correlation_matrix = correlation_matrix / (centered_activations.shape[0] - 1)
        
        # Normalize to get correlation coefficients
        std_devs = torch.diagonal(correlation_matrix).sqrt()
        correlation_matrix = correlation_matrix / (std_devs.unsqueeze(0) * std_devs.unsqueeze(1) + 1e-8)
        
        # Remove diagonal (self-correlations)
        mask = ~torch.eye(n_features, dtype=torch.bool, device=activations.device)
        off_diagonal_correlations = correlation_matrix[mask]
        
        # Encourage moderate correlations (redundancy) but not perfect correlations (diversity)
        target_correlation = self.similarity_threshold
        redundancy_loss = F.mse_loss(off_diagonal_correlations.abs(), 
                                   target_correlation * torch.ones_like(off_diagonal_correlations))
        
        return redundancy_loss


class AdversarialRobustnessLoss(nn.Module):
    """
    Loss to improve robustness against adversarial attacks on features
    """
    
    def __init__(self, epsilon: float = 0.1, attack_steps: int = 5, alpha: float = 0.01):
        super().__init__()
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.alpha = alpha
    
    def forward(
        self, 
        model: nn.Module, 
        feature_activations: List[torch.Tensor],
        safety_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial perturbations to features and ensure robustness
        """
        total_adv_loss = 0.0
        
        for layer_idx, activations in enumerate(feature_activations):
            # Generate adversarial perturbations
            perturbed_activations = self._generate_adversarial_features(
                activations, safety_scores
            )
            
            # Compute loss between original and perturbed safety scores
            # We want the safety scores to be robust to perturbations
            with torch.no_grad():
                original_safety = safety_scores
            
            # This would require access to the model's forward pass
            # For now, we'll use a simpler approximation
            perturbation_magnitude = (perturbed_activations - activations).norm(dim=-1).mean()
            
            # Encourage small perturbations to not change safety significantly
            adv_loss = perturbation_magnitude
            total_adv_loss += adv_loss
        
        return total_adv_loss / len(feature_activations)
    
    def _generate_adversarial_features(
        self, 
        activations: torch.Tensor,
        safety_scores: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial perturbations using PGD"""
        
        perturbed_activations = activations.clone().detach().requires_grad_(True)
        
        for step in range(self.attack_steps):
            if perturbed_activations.grad is not None:
                perturbed_activations.grad.zero_()
            
            # Simple loss: try to change the features
            loss = perturbed_activations.norm()
            loss.backward()
            
            # Update with gradient ascent
            with torch.no_grad():
                grad_sign = perturbed_activations.grad.sign()
                perturbed_activations += self.alpha * grad_sign
                
                # Project back to epsilon ball
                perturbation = perturbed_activations - activations
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                perturbed_activations = activations + perturbation
        
        return perturbed_activations.detach()


class ContrastiveSafetyLoss(nn.Module):
    """
    Contrastive loss to ensure safe and unsafe examples have different feature patterns
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        feature_activations: List[torch.Tensor],
        safety_labels: torch.FloatTensor
    ) -> torch.Tensor:
        """
        Args:
            feature_activations: List of [batch_size, seq_len, n_features]
            safety_labels: [batch_size] - 0 for unsafe, 1 for safe
        """
        total_contrastive_loss = 0.0
        
        for activations in feature_activations:
            # Pool features across sequence length
            pooled_features = activations.mean(dim=1)  # [batch_size, n_features]
            
            # Compute pairwise distances
            distances = torch.cdist(pooled_features, pooled_features, p=2)
            
            # Create label similarity matrix
            label_matrix = safety_labels.unsqueeze(0) == safety_labels.unsqueeze(1)
            
            # Contrastive loss
            positive_pairs = label_matrix.float()
            negative_pairs = 1.0 - positive_pairs
            
            # For positive pairs (same safety label), minimize distance
            positive_loss = positive_pairs * distances.pow(2)
            
            # For negative pairs (different safety labels), maximize distance
            negative_loss = negative_pairs * F.relu(self.margin - distances).pow(2)
            
            # Combine losses
            contrastive_loss = positive_loss + negative_loss
            
            # Exclude diagonal (self-comparisons)
            mask = ~torch.eye(len(safety_labels), dtype=torch.bool, device=activations.device)
            contrastive_loss = contrastive_loss[mask].mean()
            
            total_contrastive_loss += contrastive_loss
        
        return total_contrastive_loss / len(feature_activations) 