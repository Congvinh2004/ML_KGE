"""
TTM Negative Sampling Strategy

Negative Sampling với trustiness weights cho loss function
"""

import sys
import os

# Import base Strategy class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from openke.module.strategy.Strategy import Strategy


class TTM_NegativeSampling(Strategy):
    """
    Negative Sampling Strategy với trustiness support
    
    Tương tự NegativeSampling nhưng pass trustiness weights vào loss function
    """
    
    def __init__(self, model=None, loss=None, batch_size=256, regul_rate=0.0, l3_regul_rate=0.0):
        super(TTM_NegativeSampling, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.regul_rate = regul_rate
        self.l3_regul_rate = l3_regul_rate
    
    def _get_positive_score(self, score):
        """Lấy positive scores"""
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score
    
    def _get_negative_score(self, score):
        """Lấy negative scores"""
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score
    
    def forward(self, data):
        """
        Forward pass với trustiness support
        """
        score = self.model(data)
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        
        # Pass batch data vào loss để lấy trustiness weights
        batch_h = data['batch_h'][:self.batch_size]  # Chỉ lấy positive batch
        batch_r = data['batch_r'][:self.batch_size]
        batch_t = data['batch_t'][:self.batch_size]
        
        # Tính loss với trustiness (nếu loss function hỗ trợ)
        if hasattr(self.loss, 'forward') and len(self.loss.forward.__code__.co_varnames) > 3:
            # Loss function có thể nhận batch data
            loss_res = self.loss(p_score, n_score, batch_h, batch_r, batch_t)
        else:
            # Loss function thông thường
            loss_res = self.loss(p_score, n_score)
        
        # Regularization
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.model.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.model.l3_regularization()
        
        return loss_res

