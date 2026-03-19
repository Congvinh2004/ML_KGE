"""
TransE with Triple Trustiness Model (TransT)

Based on: Zhao et al., "Embedding Learning with Triple Trustiness on Noisy Knowledge Graph", 2019

TransT = TransE + Triple Trustiness
- Kế thừa từ TransE
- Thêm trustiness calculation
- Weighted scoring với trustiness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Import base Model class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from openke.module.model.Model import Model


class TransE_TTM(Model):
    """
    TransE with Triple Trustiness Model (TransT)
    
    Kế thừa từ TransE, thêm tính năng trustiness:
    - Tính trustiness score cho mỗi triple
    - Weighted scoring với trustiness
    - Tương thích với Trainer và Tester
    """
    
    def __init__(self, 
                 ent_tot, 
                 rel_tot, 
                 dim=100, 
                 p_norm=1, 
                 norm_flag=True, 
                 margin=None, 
                 epsilon=None,
                 trustiness_calculator=None):
        """
        Args:
            ent_tot: Tổng số entities
            rel_tot: Tổng số relations
            dim: Embedding dimension
            p_norm: L1 (1) hoặc L2 (2) norm
            norm_flag: Có normalize embeddings không
            margin: Margin cho margin loss
            epsilon: Epsilon cho initialization
            trustiness_calculator: TrustinessCalculator instance (optional)
        """
        super(TransE_TTM, self).__init__(ent_tot, rel_tot)
        
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.trustiness_calculator = trustiness_calculator
        
        # Embeddings (giống TransE)
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        
        # Initialize embeddings
        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), 
                requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False
    
    def _calc(self, h, t, r, mode):
        """
        Tính score cho triple (h, r, t) - giống TransE
        """
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score
    
    def get_trustiness(self, batch_h, batch_r, batch_t):
        """
        Lấy trustiness scores cho batch
        
        Returns:
            trustiness: Tensor of trustiness scores (shape: [batch_size])
        """
        if self.trustiness_calculator is None:
            # Nếu không có calculator, return 1.0 (default)
            return torch.ones(batch_h.shape[0], device=batch_h.device)
        
        # Convert to numpy
        h_list = batch_h.cpu().numpy() if isinstance(batch_h, torch.Tensor) else batch_h
        r_list = batch_r.cpu().numpy() if isinstance(batch_r, torch.Tensor) else batch_r
        t_list = batch_t.cpu().numpy() if isinstance(batch_t, torch.Tensor) else batch_t
        
        # Lấy trustiness cho mỗi triple
        trustiness_list = []
        for h, r, t in zip(h_list, r_list, t_list):
            trust = self.trustiness_calculator.get_trustiness(int(h), int(r), int(t))
            trustiness_list.append(trust)
        
        return torch.tensor(trustiness_list, device=batch_h.device, dtype=torch.float32)
    
    def forward(self, data):
        """
        Forward pass - giống TransE nhưng có thể thêm trustiness weighting
        """
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        
        # Lấy embeddings
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        
        # Tính score (giống TransE)
        score = self._calc(h, t, r, mode)
        
        # Có thể weight score bằng trustiness (optional)
        # if self.trustiness_calculator:
        #     trustiness = self.get_trustiness(batch_h, batch_r, batch_t)
        #     score = score * trustiness
        
        if self.margin_flag:
            return self.margin - score
        else:
            return score
    
    def regularization(self, data):
        """Regularization - giống TransE"""
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul
    
    def predict(self, data):
        """Predict - giống TransE"""
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()
    
    def set_trustiness_calculator(self, trustiness_calculator):
        """Set trustiness calculator"""
        self.trustiness_calculator = trustiness_calculator

