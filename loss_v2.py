import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OnlineKnowledgeDistillationLoss(nn.Module):
    """
    Online Knowledge Distillation Loss for cooperation between experts
    """
    def __init__(self, temperature=3.0):
        super(OnlineKnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, expert_logits):
        """
        Args:
            expert_logits: [batch_size, num_experts, num_classes]
        Returns:
            OKD loss value
        """
        batch_size, num_experts, num_classes = expert_logits.shape
        
        # Apply temperature scaling
        expert_logits_scaled = expert_logits / self.temperature
        
        # Get ensemble prediction (average of all experts)
        ensemble_logits = torch.mean(expert_logits_scaled, dim=1)  # [batch_size, num_classes]
        ensemble_probs = F.softmax(ensemble_logits, dim=1)
        
        total_loss = 0.0
        
        # Each expert learns from the ensemble
        for i in range(num_experts):
            expert_log_probs = F.log_softmax(expert_logits_scaled[:, i, :], dim=1)
            loss = self.kl_div(expert_log_probs, ensemble_probs)
            total_loss += loss
        
        return total_loss / num_experts


class ClassifierDiversificationLoss(nn.Module):
    """
    Classifier Diversification Loss to encourage different perspectives
    """
    def __init__(self):
        super(ClassifierDiversificationLoss, self).__init__()
    
    def forward(self, expert_weights):
        """
        Args:
            expert_weights: List of weight tensors from each expert's classifier
                           Each tensor has shape [num_classes, feature_dim]
        Returns:
            CD loss value (to be minimized)
        """
        num_experts = len(expert_weights)
        total_loss = 0.0
        pair_count = 0
        
        # Compare all pairs of experts
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # Normalize weights
                weight_i = F.normalize(expert_weights[i], p=2, dim=1)  # L2 normalize each class vector
                weight_j = F.normalize(expert_weights[j], p=2, dim=1)
                
                # Compute cosine similarity between corresponding class vectors
                cosine_sim = torch.sum(weight_i * weight_j, dim=1)  # [num_classes]
                
                # We want to minimize similarity (maximize diversity)
                # So we penalize high cosine similarity values
                similarity_loss = torch.mean(torch.abs(cosine_sim))
                total_loss += similarity_loss
                pair_count += 1
        
        return total_loss / pair_count if pair_count > 0 else torch.tensor(0.0)


class EnhancedCCNetLoss(nn.Module):
    """
    Combined loss function for Enhanced CCNet
    """
    def __init__(self, lambda_okd=1.0, lambda_cd=5e-8, temperature=3.0):
        super(EnhancedCCNetLoss, self).__init__()
        
        self.lambda_okd = lambda_okd
        self.lambda_cd = lambda_cd
        
        # Individual loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.okd_loss = OnlineKnowledgeDistillationLoss(temperature=temperature)
        self.cd_loss = ClassifierDiversificationLoss()
    
    def forward(self, expert_logits, expert_weights, targets):
        """
        Args:
            expert_logits: [batch_size, num_experts, num_classes]
            expert_weights: List of classifier weight tensors
            targets: [batch_size] - ground truth labels
        Returns:
            Dictionary containing individual losses and total loss
        """
        batch_size, num_experts, num_classes = expert_logits.shape
        
        # 1. Base loss: Each expert should predict correctly
        base_loss = 0.0
        for i in range(num_experts):
            expert_pred = expert_logits[:, i, :]  # [batch_size, num_classes]
            base_loss += self.ce_loss(expert_pred, targets)
        base_loss = base_loss / num_experts
        
        # 2. Online Knowledge Distillation loss: Cooperation
        okd_loss_value = self.okd_loss(expert_logits)
        
        # 3. Classifier Diversification loss: Competition/Diversity
        cd_loss_value = self.cd_loss(expert_weights)
        
        # 4. Total loss
        total_loss = base_loss + self.lambda_okd * okd_loss_value + self.lambda_cd * cd_loss_value
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'okd_loss': okd_loss_value,
            'cd_loss': cd_loss_value
        }


class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning loss (from the original CCNet)
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz]
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# Example usage and testing
if __name__ == "__main__":
    # Test the loss functions
    batch_size = 32
    num_experts = 3
    num_classes = 600
    feature_dim = 2048
    
    # Sample data
    expert_logits = torch.randn(batch_size, num_experts, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Sample expert weights (classifier weights)
    expert_weights = [torch.randn(num_classes, feature_dim) for _ in range(num_experts)]
    
    # Test individual loss functions
    okd_loss = OnlineKnowledgeDistillationLoss()
    cd_loss = ClassifierDiversificationLoss()
    
    okd_value = okd_loss(expert_logits)
    cd_value = cd_loss(expert_weights)
    
    print(f"OKD Loss: {okd_value.item():.4f}")
    print(f"CD Loss: {cd_value.item():.4f}")
    
    # Test combined loss
    combined_loss = EnhancedCCNetLoss()
    loss_dict = combined_loss(expert_logits, expert_weights, targets)
    
    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Base Loss: {loss_dict['base_loss'].item():.4f}")
    print(f"OKD Loss: {loss_dict['okd_loss'].item():.4f}")
    print(f"CD Loss: {loss_dict['cd_loss'].item():.4f}")
    print("Loss functions working correctly!")