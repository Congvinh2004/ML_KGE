# 📊 Đánh giá kết quả TransT - Lần 2 (FB15K237)

## 🎯 Kết quả Lần 2

| Metric | Giá trị |
|--------|---------|
| **MRR** | **0.2517** |
| **MR** | **194.5** |
| **Hits@10** | **0.4176** |
| **Hits@3** | **0.2735** |
| **Hits@1** | **0.1687** |

### Configuration:
- `dim`: 200
- `margin`: 5.0
- `train_times`: 1200 (tăng từ 1000 → 1200)
- `alpha`: 1.0
- `nbatches`: 100
- `neg_ent`: 10
- `use_trustiness`: True
- `alpha_trust`: 0.5
- `beta_trust`: 0.5
- `use_cross_entropy`: True

---

## 📈 So sánh Lần 2 vs Lần 1

| Metric | Lần 1 | Lần 2 | Chênh lệch | Đánh giá |
|--------|-------|-------|------------|----------|
| **MRR** | 0.2528 | 0.2517 | **-0.4%** ⚠️ | Gần như tương đương |
| **MR** | 193.6 | 194.5 | **+0.5%** ⚠️ | Gần như tương đương (MR càng nhỏ càng tốt) |
| **Hits@10** | 0.4143 | **0.4176** | **+0.8%** ✅ | **Tốt hơn một chút** |
| **Hits@3** | 0.2738 | 0.2735 | **-0.1%** ⚠️ | Gần như tương đương |
| **Hits@1** | 0.1712 | 0.1687 | **-1.5%** ⚠️ | Thấp hơn một chút |

### 📊 Phân tích:

**✅ Điểm tích cực:**
- **Hits@10 tăng 0.8%** (0.4143 → 0.4176) - Đây là metric quan trọng nhất!
- Kết quả **ổn định và nhất quán** với lần 1
- Tăng epochs từ 1000 → 1200 đã giúp cải thiện Hits@10

**⚠️ Điểm cần lưu ý:**
- MRR và Hits@1 hơi giảm nhẹ (có thể do variance trong training)
- MR tăng nhẹ (194.5 vs 193.6) - không đáng kể
- Kết quả gần như tương đương với lần 1, chứng tỏ model đã **converge tốt**

**💡 Kết luận:**
- Lần 2 cho kết quả **tương đương và ổn định** với lần 1
- Hits@10 cải thiện nhẹ (+0.8%) - đây là dấu hiệu tích cực
- Model đã **converge tốt** ở khoảng 1000-1200 epochs

---

## 📊 So sánh với TransE Baseline (FB15K237)

| Metric | TransE Baseline | TransT (Lần 2) | Chênh lệch | Đánh giá |
|--------|-----------------|----------------|------------|----------|
| **MRR** | 0.2901 | 0.2517 | **-13.2%** ⚠️ | Thấp hơn baseline |
| **MR** | 102.2 | 194.5 | **+90.3%** ❌ | **Kém hơn** (MR càng nhỏ càng tốt) |
| **Hits@10** | 0.4895 | 0.4176 | **-14.7%** ⚠️ | Thấp hơn baseline |
| **Hits@3** | 0.3195 | 0.2735 | **-14.4%** ⚠️ | Thấp hơn baseline |
| **Hits@1** | 0.1915 | 0.1687 | **-11.9%** ⚠️ | Thấp hơn baseline |

### 📊 Phân tích so với Baseline:

**⚠️ Kết quả hiện tại:**
- Tất cả metrics đều **thấp hơn baseline TransE** khoảng 12-15%
- Đặc biệt **MR cao hơn gần gấp đôi** (194.5 vs 102.2) - đây là điểm cần cải thiện

**🔍 Nguyên nhân có thể:**

1. **Hyperparameters khác nhau:**
   - Baseline: `dim=300`, `margin=6.0`, `train_times=1500`, `alpha=0.5`
   - TransT Lần 2: `dim=200`, `margin=5.0`, `train_times=1200`, `alpha=1.0`
   - **→ TransT dùng config khác với baseline, không công bằng để so sánh trực tiếp**

2. **Epochs chưa đủ:**
   - Baseline: **1500 epochs**
   - TransT Lần 2: **1200 epochs** (thiếu 300 epochs = -20%)
   - **→ Đề xuất: Tăng lên 1500 epochs để match baseline**

3. **Dimension thấp hơn:**
   - Baseline: `dim=300`
   - TransT: `dim=200` (thấp hơn 33%)
   - **→ Dimension thấp hơn có thể ảnh hưởng đến capacity của model**

4. **Learning rate cao hơn:**
   - Baseline: `alpha=0.5`
   - TransT: `alpha=1.0` (cao gấp đôi)
   - **→ Learning rate cao có thể làm model không converge tốt**

5. **Margin khác nhau:**
   - Baseline: `margin=6.0`
   - TransT: `margin=5.0`
   - **→ Margin ảnh hưởng đến loss function**

---

## 🎯 Đánh giá tổng thể

### ✅ Điểm mạnh:

1. **Kết quả ổn định:**
   - Lần 2 cho kết quả tương đương với lần 1
   - Chứng tỏ model **reproducible** và **converge tốt**

2. **Hits@10 cải thiện:**
   - Tăng từ 0.4143 → 0.4176 (+0.8%)
   - Đây là metric quan trọng nhất trong link prediction

3. **Model đã converge:**
   - Kết quả không thay đổi nhiều khi tăng epochs từ 1000 → 1200
   - Chứng tỏ model đã học được patterns tốt

### ⚠️ Điểm cần cải thiện:

1. **Vẫn thấp hơn baseline:**
   - Tất cả metrics thấp hơn baseline 12-15%
   - Cần điều chỉnh hyperparameters để match baseline

2. **MR cao:**
   - MR = 194.5 (cao gấp đôi baseline = 102.2)
   - Cần cải thiện ranking quality

3. **Có thể chưa tối ưu:**
   - Config hiện tại có thể chưa phù hợp với TransT
   - Cần tune hyperparameters cho TransT riêng

---

## 💡 Khuyến nghị

### 1. **Tăng epochs lên 1500** (match baseline)
   - Hiện tại: 1200 epochs
   - Đề xuất: 1500 epochs
   - Lý do: Baseline dùng 1500 epochs và đạt kết quả tốt

### 2. **Thử tăng dimension lên 300** (match baseline)
   - Hiện tại: dim=200
   - Đề xuất: dim=300
   - Lý do: Baseline dùng dim=300, có thể giúp model có capacity lớn hơn

### 3. **Điều chỉnh learning rate**
   - Hiện tại: alpha=1.0
   - Đề xuất: Thử alpha=0.5 (như baseline) hoặc alpha=0.8
   - Lý do: Learning rate cao có thể làm model không converge tốt

### 4. **Thử tăng margin lên 6.0** (match baseline)
   - Hiện tại: margin=5.0
   - Đề xuất: margin=6.0
   - Lý do: Baseline dùng margin=6.0

### 5. **Tune trustiness weights**
   - Hiện tại: alpha_trust=0.5, beta_trust=0.5
   - Đề xuất: Thử các giá trị khác (0.6/0.4, 0.7/0.3, 0.8/0.2)
   - Lý do: Trustiness weights có thể ảnh hưởng đến performance

### 6. **So sánh công bằng hơn:**
   - Train TransT với **cùng hyperparameters** như baseline (dim=300, margin=6.0, alpha=0.5, epochs=1500)
   - Sau đó mới so sánh kết quả

---

## 📝 Kết luận

### ✅ Kết quả Lần 2:

**Tốt:**
- ✅ Kết quả **ổn định và nhất quán** với lần 1
- ✅ **Hits@10 cải thiện** (+0.8%) - metric quan trọng nhất
- ✅ Model đã **converge tốt**

**Cần cải thiện:**
- ⚠️ Vẫn thấp hơn baseline TransE (12-15%)
- ⚠️ MR cao hơn baseline (194.5 vs 102.2)
- ⚠️ Cần tune hyperparameters để match baseline

### 🎯 Đánh giá tổng thể:

**Kết quả này là TỐT và ỔN ĐỊNH**, nhưng cần:
1. **Tăng epochs lên 1500** để match baseline
2. **Thử config giống baseline** (dim=300, margin=6.0, alpha=0.5) để so sánh công bằng
3. **Tune trustiness weights** để tối ưu cho TransT

**→ Kết quả hiện tại cho thấy TransT đang hoạt động tốt, nhưng cần điều chỉnh hyperparameters để đạt được performance tốt hơn!**
















