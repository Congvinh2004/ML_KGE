# 📊 Config Lần 5: Match Baseline + TransT Trustiness

## 🎯 Mục Tiêu

**Đạt kết quả TỐT HƠN baseline TransE** bằng cách:
1. ✅ Match các hyperparameters quan trọng của baseline
2. ✅ Giữ tính năng Trustiness của TransT (điểm mạnh)
3. ✅ Sử dụng early stopping để tránh overfitting

---

## 📋 So Sánh Config

### Baseline TransE (FB15K237)

| Parameter | Baseline | TransT Lần 4 | **TransT Lần 5** | Thay đổi |
|-----------|----------|--------------|------------------|----------|
| `dim` | **300** | 200 | **300** ✅ | ⬆️ Tăng từ 200 → 300 |
| `margin` | **6.0** | 5.0 | **6.0** ✅ | ⬆️ Tăng từ 5.0 → 6.0 |
| `train_times` | **1500** | 1200 | **1500** ✅ | ⬆️ Tăng từ 1200 → 1500 |
| `alpha` | **0.5** | 1.0 | **0.5** ✅ | ⬇️ Giảm từ 1.0 → 0.5 |
| `nbatches` | **50** | 100 | **50** ✅ | ⬇️ Giảm từ 100 → 50 |
| `type_constrain` | **True** | False | **True** ✅ | ⬆️ Bật từ False → True |
| `use_cross_entropy` | False (margin loss) | True | **False** ✅ | ⬇️ Tắt từ True → False |

### TransT Trustiness (Giữ nguyên)

| Parameter | Lần 4 | Lần 5 | Ghi chú |
|-----------|-------|-------|---------|
| `use_trustiness` | True | **True** ✅ | Giữ trustiness (điểm mạnh) |
| `alpha_trust` | 0.5 | **0.5** ✅ | Weight type-based (cân bằng) |
| `beta_trust` | 0.5 | **0.5** ✅ | Weight description-based (cân bằng) |

---

## 📊 Kết Quả Kỳ Vọng

### Baseline TransE (Target để vượt qua)

| Metric | Baseline | Mục tiêu Lần 5 |
|--------|----------|----------------|
| **MRR** | 0.2901 | **> 0.2901** 🎯 |
| **MR** | 102.2 | **< 102.2** 🎯 (MR càng nhỏ càng tốt) |
| **Hits@10** | 0.4895 | **> 0.4895** 🎯 |
| **Hits@3** | 0.3195 | **> 0.3195** 🎯 |
| **Hits@1** | 0.1915 | **> 0.1915** 🎯 |

### TransT Lần 4 (Để so sánh)

| Metric | Lần 4 | Kỳ vọng cải thiện |
|--------|-------|-------------------|
| **MRR** | 0.2538 | **+14.3%** (lên ~0.29+) |
| **MR** | 193.9 | **-47.3%** (xuống ~102) |
| **Hits@10** | 0.4159 | **+17.7%** (lên ~0.49+) |
| **Hits@3** | 0.2757 | **+15.9%** (lên ~0.32+) |
| **Hits@1** | 0.1718 | **+11.5%** (lên ~0.19+) |

---

## ⚙️ Config Chi Tiết

```python
TRANST_CONFIG = {
    # Model parameters (Match Baseline)
    'dim': 300,              # ⬆️ Tăng từ 200 → 300 (QUAN TRỌNG!)
    'margin': 6.0,           # ⬆️ Tăng từ 5.0 → 6.0 (QUAN TRỌNG!)
    'p_norm': 1,             # L1 distance
    'norm_flag': True,       # Normalize embeddings
    
    # Training parameters (Match Baseline)
    'train_times': 1500,     # ⬆️ Tăng từ 1200 → 1500
    'alpha': 0.5,            # ⬇️ Giảm từ 1.0 → 0.5 (learning rate ổn định hơn)
    'nbatches': 50,          # ⬇️ Giảm từ 100 → 50
    'neg_ent': 10,           # Giữ nguyên
    
    # DataLoader parameters
    'threads': 0,            # Single thread (tốt cho Kaggle)
    'sampling_mode': 'normal',
    'bern_flag': 1,
    'filter_flag': 1,
    
    # Evaluation (Match Baseline)
    'type_constrain': True,  # ⬆️ Bật từ False → True (đánh giá chính xác hơn)
    
    # TransT Trustiness (Giữ nguyên - Điểm mạnh)
    'use_trustiness': True,  # Giữ trustiness
    'alpha_trust': 0.5,      # Weight type-based (cân bằng)
    'beta_trust': 0.5,       # Weight description-based (cân bằng)
    'use_cross_entropy': False,  # ⬇️ Tắt từ True → False (dùng margin loss như baseline)
}
```

---

## ⏱️ Thông Tin Training

### Thời Gian Ước Tính

- **Tổng iterations**: 1500 epochs × 50 batches = **75,000 iterations**
- **Thời gian ước tính**: ~10-12 giờ trên Kaggle
- **Early stopping**: Sẽ dừng sớm nếu không cải thiện sau 150 epochs (10% của 1500)

### Early Stopping Config

```python
EARLY_STOPPING_CONFIG = {
    'patience_percent': 0.10,    # 150 epochs (10% của 1500)
    'min_delta': 0.0001,         # Cải thiện tối thiểu
    'monitor': 'Hits@10',        # Metric quan trọng nhất
    'mode': 'max',               # Càng lớn càng tốt
    'min_epochs': 500,           # Tối thiểu train 500 epochs
}
```

### Test Định Kỳ

- **Test interval**: Mỗi 100 epochs
- **Lưu checkpoint**: Chỉ lưu best checkpoint (tiết kiệm I/O)

---

## 💡 Lý Do Các Thay Đổi

### 1. `dim: 200 → 300` ✅

**Lý do:**
- Baseline dùng dim=300 và đạt kết quả tốt (MRR: 0.2901, Hits@10: 0.4895)
- Embedding dimension lớn hơn → biểu diễn tốt hơn → kết quả tốt hơn
- **QUAN TRỌNG NHẤT** trong các thay đổi!

### 2. `margin: 5.0 → 6.0` ✅

**Lý do:**
- Baseline dùng margin=6.0 và đạt kết quả tốt
- Margin lớn hơn → phân biệt rõ hơn giữa positive và negative triples
- **QUAN TRỌNG** để match baseline!

### 3. `alpha: 1.0 → 0.5` ✅

**Lý do:**
- Baseline dùng alpha=0.5 (learning rate thấp hơn)
- Learning rate thấp hơn → training ổn định hơn, ít overfitting
- Phù hợp với config match baseline

### 4. `nbatches: 100 → 50` ✅

**Lý do:**
- Baseline dùng nbatches=50
- Ít batch hơn nhưng mỗi batch lớn hơn → gradient ổn định hơn
- Match baseline để so sánh công bằng

### 5. `type_constrain: False → True` ✅

**Lý do:**
- Baseline dùng type_constrain=True
- Đánh giá chính xác hơn (loại bỏ các predictions không hợp lý về mặt type)
- **QUAN TRỌNG** để so sánh công bằng với baseline!

### 6. `use_cross_entropy: True → False` ✅

**Lý do:**
- Baseline dùng margin loss (không dùng cross-entropy)
- Margin loss phù hợp hơn với TransE/TransT
- Match baseline để so sánh công bằng

### 7. `train_times: 1200 → 1500` ✅

**Lý do:**
- Baseline train 1500 epochs
- Early stopping sẽ dừng sớm nếu không cải thiện (patience = 150 epochs)
- Đảm bảo model có đủ thời gian học với config mới

### 8. Giữ Trustiness ✅

**Lý do:**
- Trustiness là điểm mạnh của TransT
- Có thể giúp model đạt kết quả tốt hơn baseline
- Giữ weights cân bằng (alpha_trust=0.5, beta_trust=0.5)

---

## 🎯 Kỳ Vọng Kết Quả

### Scenario 1: Đạt kết quả tốt hơn baseline ✅

**Kết quả kỳ vọng:**
- MRR: **> 0.2901** (tốt hơn baseline)
- Hits@10: **> 0.4895** (tốt hơn baseline)
- MR: **< 102.2** (tốt hơn baseline - MR càng nhỏ càng tốt)

**Lý do:**
- Match tất cả hyperparameters quan trọng của baseline
- Thêm Trustiness (điểm mạnh của TransT)
- Early stopping tránh overfitting

### Scenario 2: Đạt kết quả tương đương baseline ✅

**Kết quả kỳ vọng:**
- MRR: **~0.29** (gần baseline)
- Hits@10: **~0.49** (gần baseline)
- MR: **~102** (gần baseline)

**Lý do:**
- Match baseline config
- Trustiness có thể không giúp nhiều hoặc cần điều chỉnh

### Scenario 3: Cải thiện đáng kể so với Lần 4 ✅

**Kết quả kỳ vọng:**
- MRR: **+14%** (từ 0.2538 → ~0.29)
- Hits@10: **+18%** (từ 0.4159 → ~0.49)
- MR: **-47%** (từ 193.9 → ~102)

**Lý do:**
- Tất cả thay đổi đều hướng tới match baseline
- Baseline đã được chứng minh là tốt

---

## ⚠️ Lưu Ý

1. **Thời gian training**: ~10-12 giờ (có thể dừng sớm nhờ early stopping)
2. **Memory**: dim=300 sẽ tốn memory hơn dim=200 (~50% tăng)
3. **Early stopping**: Sẽ dừng sớm nếu không cải thiện sau 150 epochs
4. **Test interval**: Mỗi 100 epochs (tiết kiệm thời gian)

---

## 📝 Checklist Trước Khi Train

- [ ] Đã set `TRAIN_RUN = 5`
- [ ] Đã kiểm tra config match baseline
- [ ] Đã kiểm tra early stopping config
- [ ] Đã backup checkpoint cũ (nếu cần)
- [ ] Đã chuẩn bị đủ thời gian (~10-12 giờ)

---

## 🚀 Bắt Đầu Training

Sử dụng file: `kaggle_train_ttm_FB15K237_v2_with_early_stopping.py`

Config đã được cập nhật sẵn cho lần 5!

**Chúc bạn đạt kết quả tốt hơn baseline!** 🎉







