"""
Custom TransE Model - Có thể customize và cải tiến

Mô hình TransE với các tính năng mở rộng:
- Dropout để tránh overfitting
- Batch Normalization để ổn định training
- Flexible initialization methods
- Có thể thêm attention, regularization, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Import base Model từ OpenKE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from openke.module.model.Model import Model


class TransE_Custom(Model):
    """
    TransE Custom Model - Cải tiến từ TransE gốc
    
    Features:
    - Dropout layer (optional)
    - Batch Normalization (optional)
    - Flexible initialization
    - Customizable scoring function
    
    Args:
        ent_tot: Tổng số entities
        rel_tot: Tổng số relations
        dim: Embedding dimension (default: 100)
        p_norm: Norm type cho distance (1 hoặc 2, default: 1)
        norm_flag: Có normalize embeddings không (default: True)
        margin: Margin cho margin loss (default: None)
        epsilon: Epsilon cho initialization (default: None)
        dropout: Dropout rate (0.0 = không dùng, default: 0.0)
        use_batch_norm: Có dùng BatchNorm không (default: False)
        init_method: Phương thức khởi tạo ('xavier', 'uniform', 'normal', default: 'xavier')
    """
    
    def __init__(self, 
                 ent_tot, 
                 rel_tot, 
                 dim=100, 
                 p_norm=1, 
                 norm_flag=True, 
                 margin=None, 
                 epsilon=None,
                 dropout=0.0,
                 use_batch_norm=False,
                 init_method='xavier'):
        super(TransE_Custom, self).__init__(ent_tot, rel_tot)
        
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        
        # Embedding layers
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        
        # Optional: Dropout layer
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        # Optional: Batch Normalization (thực ra dùng LayerNorm cho embeddings)
        if use_batch_norm:
            # LayerNorm hoạt động tốt hơn cho embeddings so với BatchNorm
            self.ent_bn = nn.LayerNorm(self.dim, elementwise_affine=False)
            self.rel_bn = nn.LayerNorm(self.dim, elementwise_affine=False)
        else:
            self.ent_bn = None
            self.rel_bn = None
        
        # Initialize embeddings
        self._init_embeddings(init_method)
        
        # Margin flag
        if margin is not None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False
    
    def _init_embeddings(self, init_method):
        """Khởi tạo embeddings theo phương thức được chọn"""
        if self.margin is not None and self.epsilon is not None:
            # Uniform initialization với margin và epsilon
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
        else:
            # Các phương thức khởi tạo khác
            if init_method == 'xavier':
                nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
                nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            elif init_method == 'normal':
                nn.init.normal_(self.ent_embeddings.weight.data, mean=0.0, std=0.1)
                nn.init.normal_(self.rel_embeddings.weight.data, mean=0.0, std=0.1)
            elif init_method == 'uniform':
                nn.init.uniform_(self.ent_embeddings.weight.data, a=-0.1, b=0.1)
                nn.init.uniform_(self.rel_embeddings.weight.data, a=-0.1, b=0.1)
            else:
                # Default: xavier
                nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
                nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
    
    def _apply_dropout_bn(self, embeddings, bn_layer):
        """Áp dụng dropout và batch normalization nếu được bật"""
        # Dropout - áp dụng trước normalization
        if self.dropout is not None and self.dropout_rate > 0.0:
            embeddings = self.dropout(embeddings)
        
        # Batch normalization
        # Lưu ý: BatchNorm trong KGE thường không hiệu quả như trong CNN
        # Nên chỉ dùng với cẩn thận. Ở đây dùng LayerNorm thay vì BatchNorm
        if bn_layer is not None and self.use_batch_norm:
            # Reshape để normalization hoạt động trên dimension cuối
            original_shape = embeddings.shape
            if len(original_shape) > 2:
                embeddings = embeddings.view(-1, embeddings.shape[-1])
            
            embeddings = bn_layer(embeddings)
            
            # Reshape lại
            if len(original_shape) > 2:
                embeddings = embeddings.view(original_shape)
        
        return embeddings
    
    def _calc(self, h, t, r, mode):
        """
        Tính score cho triplet (h, r, t)
        
        Score function: ||h + r - t||_{L_p}
        """
        # Apply dropout và batch norm nếu có
        h = self._apply_dropout_bn(h, self.ent_bn)
        t = self._apply_dropout_bn(t, self.ent_bn)
        r = self._apply_dropout_bn(r, self.rel_bn)
        
        # Normalize embeddings nếu được bật
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        
        # Reshape cho batch processing
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        
        # Tính score: TransE scoring function
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        
        # L1 hoặc L2 norm
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: Dictionary chứa:
                - batch_h: head entities
                - batch_t: tail entities
                - batch_r: relations
                - mode: 'normal', 'head_batch', hoặc 'tail_batch'
        """
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        
        # Lấy embeddings
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        
        # Tính score
        score = self._calc(h, t, r, mode)
        
        # Return margin - score nếu dùng margin loss
        if self.margin_flag:
            return self.margin - score
        else:
            return score
    
    def regularization(self, data):
        """
        Regularization term để tránh overfitting
        L2 regularization trên embeddings
        """
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        
        # L2 regularization
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul
    
    def predict(self, data):
        """
        Dự đoán score cho triplets
        """
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
        return score.cpu().data.numpy()


# Ví dụ sử dụng:
"""
from research.models.TransE_Custom import TransE_Custom

# Mô hình TransE cơ bản (giống TransE gốc)
model = TransE_Custom(
    ent_tot=14951,
    rel_tot=237,
    dim=200,
    p_norm=1,
    norm_flag=True,
    margin=5.0
)

# Mô hình TransE với Dropout
model_dropout = TransE_Custom(
    ent_tot=14951,
    rel_tot=237,
    dim=200,
    p_norm=1,
    norm_flag=True,
    margin=5.0,
    dropout=0.1  # 10% dropout
)

# Mô hình TransE với Batch Normalization
model_bn = TransE_Custom(
    ent_tot=14951,
    rel_tot=237,
    dim=200,
    p_norm=1,
    norm_flag=True,
    margin=5.0,
    use_batch_norm=True
)

# Mô hình TransE với cả Dropout và BatchNorm
model_full = TransE_Custom(
    ent_tot=14951,
    rel_tot=237,
    dim=200,
    p_norm=1,
    norm_flag=True,
    margin=5.0,
    dropout=0.1,
    use_batch_norm=True,
    init_method='normal'
)
"""
