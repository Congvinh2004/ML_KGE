# 🎯 Triple Trustiness Model (TTM)

**Triple Trustiness Model** - TransE với Triple Trustiness để xử lý dữ liệu nhiễu trong Knowledge Graph.

**Based on:** Zhao et al., "Embedding Learning with Triple Trustiness on Noisy Knowledge Graph", 2019

---

## 📁 Cấu trúc thư mục

```
research/ttm/
├── __init__.py                    # Main module exports
├── README.md                      # File này
│
├── models/                        # TTM Models
│   ├── __init__.py
│   ├── TransE_TTM.py             # TransE với Triple Trustiness
│   └── TrustinessCalculator.py   # Tính trustiness scores
│
├── loss/                          # TTM Loss Functions
│   ├── __init__.py
│   └── TTM_Loss.py               # Cross-entropy loss với trustiness weights
│
├── strategy/                      # TTM Training Strategies
│   ├── __init__.py
│   └── TTM_NegativeSampling.py   # Negative sampling với trustiness
│
└── experiments/                   # Training Scripts
    ├── train_transe_ttm.py       # Script train TransT trên WN18RR
    └── train_transe_ttm_fb15k237.py  # Script train TransT trên FB15K237
```

---

## 🚀 Quick Start

### Import TTM

```python
from research.ttm import TransE_TTM, TrustinessCalculator, TTM_Loss, TTM_NegativeSampling
```

### Hoặc import từng module

```python
from research.ttm.models import TransE_TTM, TrustinessCalculator
from research.ttm.loss import TTM_Loss
from research.ttm.strategy import TTM_NegativeSampling
```

---

## 📖 Sử dụng

### 1. Tính Trustiness

```python
from research.ttm import TrustinessCalculator

# Tạo calculator
calculator = TrustinessCalculator(
    dataset_path="./benchmarks/WN18RR/",
    alpha=0.5,  # Weight cho type-based
    beta=0.5    # Weight cho description-based
)

# Load entity types và descriptions
calculator.load_entity_types()
calculator.load_entity_descriptions()

# Tính trustiness cho triple
trustiness = calculator.get_trustiness(h=0, r=1, t=2)
```

### 2. Tạo TransT Model

```python
from research.ttm import TransE_TTM

transt = TransE_TTM(
    ent_tot=40943,
    rel_tot=11,
    dim=100,
    p_norm=1,
    norm_flag=True,
    margin=6.0,
    trustiness_calculator=calculator  # Optional
)
```

### 3. Tạo TTM Loss

```python
from research.ttm import TTM_Loss

ttm_loss = TTM_Loss(
    margin=6.0,
    use_cross_entropy=True,  # Cross-entropy (True) hay margin loss (False)
    trustiness_weights=calculator.trustiness_scores
)
```

### 4. Train Model

```python
from research.ttm import TTM_NegativeSampling
from openke.config import Trainer

# Tạo strategy
model = TTM_NegativeSampling(
    model=transt,
    loss=ttm_loss,
    batch_size=256
)

# Train
trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=2000,
    alpha=0.1,
    use_gpu=True
)

trainer.run()
```

---

## 📝 Chi tiết Components

### TransE_TTM

Kế thừa từ `TransE`, thêm:
- Trustiness calculator integration
- Weighted scoring với trustiness

### TrustinessCalculator

Tính trustiness từ:
1. **Type-based**: Entity type instances
2. **Description-based**: Entity descriptions

Formula:
```
trustiness(h,r,t) = α * type_trustiness + β * desc_trustiness
```

### TTM_Loss

Loss function với trustiness weights:
- **Cross-entropy**: `L = -Σ trustiness(h,r,t) * log σ(score(h,r,t))`
- **Margin loss**: `L = Σ trustiness(h,r,t) * max(0, margin - score(h,r,t))`

### TTM_NegativeSampling

Negative sampling strategy với trustiness support.

---

## 🔧 Configuration

### WN18RR Configuration

```python
TRANST_CONFIG = {
    # Standard TransE
    'dim': 100,              # Embedding dimension
    'margin': 6.0,           # Margin for loss
    'train_times': 2000,     # Number of epochs
    'alpha': 0.1,            # Learning rate
    'nbatches': 50,          # Number of batches per epoch
    'neg_ent': 5,            # Negative samples per positive
    
    # TTM specific
    'use_trustiness': True,
    'alpha_trust': 0.5,      # Type-based weight
    'beta_trust': 0.5,        # Description-based weight
    'use_cross_entropy': True, # Cross-entropy (True) hay margin (False)
}
```

### FB15K237 Configuration

```python
TRANST_CONFIG = {
    # Standard TransE (khác với WN18RR)
    'dim': 200,              # Embedding dimension (lớn hơn WN18RR)
    'margin': 5.0,           # Margin for loss
    'train_times': 1000,     # Number of epochs
    'alpha': 1.0,            # Learning rate (cao hơn WN18RR)
    'nbatches': 100,         # Number of batches per epoch (nhiều hơn)
    'neg_ent': 10,           # Negative samples per positive (nhiều hơn)
    
    # TTM specific
    'use_trustiness': True,
    'alpha_trust': 0.5,      # Type-based weight
    'beta_trust': 0.5,        # Description-based weight
    'use_cross_entropy': True, # Cross-entropy (True) hay margin (False)
}
```

**Lưu ý:** FB15K237 thường cần hyperparameters khác WN18RR:
- Dimension lớn hơn (200 vs 100)
- Learning rate cao hơn (1.0 vs 0.1)
- Nhiều batches hơn (100 vs 50)
- Nhiều negative samples hơn (10-25 vs 5)

---

## 📊 So sánh với TransE

| Aspect | TransE | TransT (TTM) |
|--------|--------|--------------|
| **Model** | TransE | TransE + Trustiness |
| **Loss** | Margin Loss | Cross-entropy với weights |
| **Noise Handling** | Không | Có (weighted by trustiness) |
| **Trustiness** | Không | Có (type + description) |

---

## 📚 Tài liệu tham khảo

- **Paper**: Zhao et al., "Embedding Learning with Triple Trustiness on Noisy Knowledge Graph", 2019
- **OpenKE**: https://github.com/thunlp/OpenKE

---

## 🚀 Training Scripts

### Train trên WN18RR

```bash
python research/ttm/experiments/train_transe_ttm.py
```

### Train trên FB15K237

```bash
python research/ttm/experiments/train_transe_ttm_fb15k237.py
```

**Checkpoints sẽ được lưu tại:**
- WN18RR: `research/ttm/checkpoints/transt_wn18rr.ckpt`
- FB15K237: `research/ttm/checkpoints/transt_fb15k237.ckpt`

**Kết quả sẽ được lưu tại:**
- FB15K237: `research/ttm/checkpoints/transt_fb15k237_results.txt`

---

## ✅ Checklist

- [x] TransE_TTM model
- [x] TrustinessCalculator
- [x] TTM_Loss function
- [x] TTM_NegativeSampling strategy
- [x] Training script cho WN18RR
- [x] Training script cho FB15K237
- [x] Test trên WN18RR
- [ ] Test trên FB15K237
- [x] So sánh với TransE baseline (WN18RR)
- [ ] So sánh với TransE baseline (FB15K237)

---

**Version:** 1.0.0  
**Last Updated:** 2024

