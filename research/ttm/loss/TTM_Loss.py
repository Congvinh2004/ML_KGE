"""
TTM Loss Function - Cross-entropy based loss with trustiness weights

Based on: Zhao et al., "Embedding Learning with Triple Trustiness on Noisy Knowledge Graph", 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Import base Loss class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from openke.module.loss.Loss import Loss


class TTM_Loss(Loss):
    """
    Triple Trustiness Model Loss Function
    
    Cross-entropy based loss với trustiness weights:
    L = -Σ trustiness(h,r,t) * log σ(score(h,r,t))
    
    Hoặc weighted margin loss:
    L = Σ trustiness(h,r,t) * max(0, margin - score(h,r,t))
    """
    
    def __init__(self, margin=6.0, use_cross_entropy=True, trustiness_weights=None):
        """
        Args:
            margin: Margin cho margin loss (nếu không dùng cross-entropy)
            use_cross_entropy: Dùng cross-entropy loss (True) hay margin loss (False)
            trustiness_weights: Dict {(h,r,t): trustiness_score} hoặc None
        """
        super(TTM_Loss, self).__init__()
        
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.use_cross_entropy = use_cross_entropy
        self.trustiness_weights = trustiness_weights or {}
        
        # Sigmoid function cho cross-entropy
        self.sigmoid = nn.Sigmoid()
        
    def get_trustiness_weight(self, batch_h, batch_r, batch_t):
        """
        Lấy trustiness weights cho batch
        
        Args:
            batch_h: Head entities (tensor)
            batch_r: Relations (tensor)
            batch_t: Tail entities (tensor)
        
        Returns:
            trustiness_weights: Tensor of trustiness scores
        """
        if not self.trustiness_weights:
            # Nếu không có trustiness weights, dùng 1.0 (default)
            return torch.ones(batch_h.shape[0], device=batch_h.device)
        
        # Convert to numpy nếu cần
        if isinstance(batch_h, torch.Tensor):
            h_list = batch_h.cpu().numpy()
            r_list = batch_r.cpu().numpy()
            t_list = batch_t.cpu().numpy()
        else:
            h_list = batch_h
            r_list = batch_r
            t_list = batch_t
        
        # Lấy trustiness cho mỗi triple trong batch
        weights = []
        for h, r, t in zip(h_list, r_list, t_list):
            trust = self.trustiness_weights.get((int(h), int(r), int(t)), 1.0)
            weights.append(trust)
        
        return torch.tensor(weights, device=batch_h.device, dtype=torch.float32)
    
    def forward(self, p_score, n_score, batch_h=None, batch_r=None, batch_t=None):
        """
        Forward pass với trustiness weights
        
        Args:
            p_score: Positive scores (tensor)
            n_score: Negative scores (tensor)
            batch_h, batch_r, batch_t: Batch data để lấy trustiness weights
        
        Returns:
            loss: Weighted loss value
        """
        if self.use_cross_entropy:
            # Cross-entropy based loss với trustiness weights
            return self._cross_entropy_loss(p_score, n_score, batch_h, batch_r, batch_t)
        else:
            # Weighted margin loss
            return self._margin_loss(p_score, n_score, batch_h, batch_r, batch_t)
    
    def _cross_entropy_loss(self, p_score, n_score, batch_h, batch_r, batch_t):
        """
        Cross-entropy loss với trustiness weights:
        L = -Σ trustiness(h,r,t) * log σ(score(h,r,t))
        """
        # Tính probability cho positive và negative
        p_prob = self.sigmoid(p_score)  # P((h,r,t) is true)
        n_prob = self.sigmoid(-n_score)  # P((h',r,t) is false)
        
        # Cross-entropy loss
        p_loss = -torch.log(p_prob + 1e-10)  # Loss cho positive
        n_loss = -torch.log(n_prob + 1e-10)  # Loss cho negative
        
        # Nếu có trustiness weights
        if batch_h is not None and self.trustiness_weights:
            trustiness = self.get_trustiness_weight(batch_h, batch_r, batch_t)
            # Weighted loss
            p_loss = trustiness.view(-1, 1) * p_loss
            n_loss = trustiness.view(-1, 1) * n_loss
        
        # Average loss
        loss = (p_loss.mean() + n_loss.mean()) / 2
        
        return loss
    
    def _margin_loss(self, p_score, n_score, batch_h, batch_r, batch_t):
        """
        Weighted margin loss:
        L = Σ trustiness(h,r,t) * max(0, margin - score(h,r,t))
        """
        # Margin loss
        margin_loss = torch.max(p_score - n_score, -self.margin) + self.margin
        
        # Nếu có trustiness weights
        if batch_h is not None and self.trustiness_weights:
            trustiness = self.get_trustiness_weight(batch_h, batch_r, batch_t)
            # Weighted loss
            margin_loss = trustiness.view(-1, 1) * margin_loss
        
        return margin_loss.mean()
    
    def set_trustiness_weights(self, trustiness_weights):
        """Set trustiness weights"""
        self.trustiness_weights = trustiness_weights
    
    def predict(self, p_score, n_score):
        """Predict (for compatibility)"""
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()

