# 📊 So sánh Config TransT cho FB15K237

## 📈 Kết quả hiện tại (Config Version 1)

| Metric | TransT (V1) | Baseline TransE | Chênh lệch |
|--------|------------|----------------|------------|
| **MRR** | 0.2528 | 0.2901 | -12.9% ⚠️ |
| **MR** | 193.6 | 102.2 | +89.4% ❌ |
| **Hits@10** | 0.4143 | 0.4895 | -15.4% ⚠️ |
| **Hits@3** | 0.2738 | 0.3195 | -14.3% ⚠️ |
| **Hits@1** | 0.1712 | 0.1915 | -10.6% ⚠️ |

**Config V1:**
- dim: 200
- margin: 5.0
- train_times: 1000
- alpha: 1.0
- nbatches: 100
- use_cross_entropy: True
- alpha_trust: 0.5, beta_trust: 0.5

---

## 🎯 Config Version 2: Match Baseline (Recommended)

**Mục tiêu:** Match các hyperparameters của baseline để cải thiện kết quả

### Thay đổi so với V1:

| Parameter | V1 | V2 | Lý do |
|-----------|----|----|-------|
| `dim` | 200 | **300** | Tăng để match baseline (300) |
| `margin` | 5.0 | **6.0** | Tăng để match baseline (6.0) |
| `train_times` | 1000 | **1500** | Tăng epochs để match baseline |
| `alpha` | 1.0 | **0.5** | Giảm learning rate để match baseline |
| `nbatches` | 100 | **50** | Giảm để match baseline |
| `type_constrain` | False | **True** | Bật type constraint như baseline |
| `use_cross_entropy` | True | **False** | Dùng margin loss như baseline |
| `alpha_trust` | 0.5 | **0.3** | Giảm weight type-based |
| `beta_trust` | 0.5 | **0.7** | Tăng weight description-based |

### Config V2:

```python
TRANST_CONFIG = {
    'dim': 300,
    'margin': 6.0,
    'train_times': 1500,
    'alpha': 0.5,
    'nbatches': 50,
    'neg_ent': 10,
    'threads': 0,
    'p_norm': 1,
    'norm_flag': True,
    'type_constrain': True,
    
    # TTM specific
    'use_trustiness': True,
    'alpha_trust': 0.3,
    'beta_trust': 0.7,
    'use_cross_entropy': False,
}
```

**⏱️ Thời gian train:** ~13-14 giờ trên Kaggle

**💡 Kỳ vọng:** Cải thiện tất cả metrics, đặc biệt là MRR và Hits@10

---

## ⚖️ Config Version 3: Balanced (Alternative)

**Mục tiêu:** Cân bằng giữa thời gian train và chất lượng

### Thay đổi so với V2:

| Parameter | V2 | V3 | Lý do |
|-----------|----|----|-------|
| `train_times` | 1500 | **1200** | Giảm để train nhanh hơn (~10-11 giờ) |
| `nbatches` | 50 | **100** | Tăng để train nhanh hơn |
| `alpha_trust` | 0.3 | **0.4** | Cân bằng hơn |
| `beta_trust` | 0.7 | **0.6** | Cân bằng hơn |
| `use_cross_entropy` | False | **True** | Thử cross-entropy với trustiness |

### Config V3:

```python
TRANST_CONFIG = {
    'dim': 300,
    'margin': 6.0,
    'train_times': 1200,      # Giảm từ 1500
    'alpha': 0.5,
    'nbatches': 100,          # Tăng từ 50
    'neg_ent': 10,
    'threads': 0,
    'p_norm': 1,
    'norm_flag': True,
    'type_constrain': True,
    
    # TTM specific
    'use_trustiness': True,
    'alpha_trust': 0.4,       # Cân bằng hơn
    'beta_trust': 0.6,        # Cân bằng hơn
    'use_cross_entropy': True,  # Thử cross-entropy
}
```

**⏱️ Thời gian train:** ~10-11 giờ trên Kaggle

**💡 Kỳ vọng:** Kết quả tốt với thời gian train ngắn hơn

---

## 📋 Baseline TransE (Reference)

```python
BASELINE_CONFIG = {
    'dim': 300,
    'margin': 6.0,
    'train_times': 1500,
    'alpha': 0.5,
    'nbatches': 50,
    'neg_ent': 10,
    'type_constrain': True,
}
```

**Kết quả Baseline:**
- MRR: 0.2901
- MR: 102.2
- Hits@10: 0.4895
- Hits@3: 0.3195
- Hits@1: 0.1915

---

## 🎯 Khuyến nghị

1. **Bắt đầu với Config V2** (Match Baseline)
   - Có khả năng cao nhất để cải thiện kết quả
   - Match các hyperparameters quan trọng với baseline

2. **Nếu thời gian train quá lâu, dùng Config V3** (Balanced)
   - Train nhanh hơn ~2-3 giờ
   - Vẫn giữ các hyperparameters quan trọng

3. **Sau khi có kết quả V2, thử nghiệm:**
   - Điều chỉnh `alpha_trust` và `beta_trust`
   - Thử `use_cross_entropy: True` với V2 config
   - Thử tăng `neg_ent` lên 15-20

---

## 📝 Ghi chú

- **MR (Mean Rank):** Càng nhỏ càng tốt
- **MRR (Mean Reciprocal Rank):** Càng lớn càng tốt
- **Hits@K:** Càng lớn càng tốt
- **type_constrain:** Bật sẽ cho kết quả tốt hơn nhưng chậm hơn khi test
- **use_cross_entropy:** Cross-entropy có thể tốt hơn với trustiness mechanism
























