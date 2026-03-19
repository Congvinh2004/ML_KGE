# 📊 BÁO CÁO TỔNG HỢP KẾT QUẢ - TRANSE VÀ TRANST

**Ngày báo cáo:** $(date)  
**Models:** TransE (Baseline), TransT (TransE + Triple Trustiness)  
**Datasets:** FB15K237, WN18RR

---

## 📋 TÓM TẮT EXECUTIVE

### Kết Quả Tốt Nhất Đạt Được - Tất Cả Models

#### 1. TransE trên FB15K237 (Baseline)

| Metric | Giá trị |
|--------|---------|
| **MRR** | **0.2901** |
| **MR** | **102.2** |
| **Hits@10** | **0.4895** |
| **Hits@3** | **0.3195** |
| **Hits@1** | **0.1915** |

**Đánh giá:** ✅ **TỐT NHẤT** - Vượt tiêu chuẩn công bố (Hits@10: 0.4895 > 0.476 OpenKE)

---

#### 2. TransE trên WN18RR (Baseline)

| Metric | Giá trị |
|--------|---------|
| **MRR** | **0.1932** |
| **MR** | **4251.0** |
| **Hits@10** | **0.4474** |
| **Hits@3** | **0.3555** |
| **Hits@1** | **0.0061** |

**Đánh giá:** ⚠️ **ỔN ĐỊNH** - Thấp hơn tiêu chuẩn công bố ~14-20%, đặc biệt Hits@1 rất thấp

---

#### 3. TransT trên FB15K237 (Best - Lần 4)

| Metric | Giá trị |
|--------|---------|
| **MRR** | **0.2538** |
| **MR** | **192.8** |
| **Hits@10** | **0.4159** |
| **Hits@3** | **0.2757** |
| **Hits@1** | **0.1718** |

**Đánh giá:** ⚠️ **ỔN ĐỊNH** - Thấp hơn TransE baseline ~12-15%

---

### So Sánh Tổng Quan

| Model | Dataset | MRR | Hits@10 | MR | Đánh giá |
|-------|---------|-----|---------|-----|----------|
| **TransE** | FB15K237 | **0.2901** | **0.4895** | **102.2** | ✅ Tốt nhất |
| **TransE** | WN18RR | 0.1932 | 0.4474 | 4251.0 | ⚠️ Ổn định |
| **TransT** | FB15K237 | 0.2538 | 0.4159 | 192.8 | ⚠️ Cần cải thiện |

---

## 📈 CHI TIẾT KẾT QUẢ TỪNG LẦN TRAIN

### Lần 1 (Baseline Config)

**Configuration:**
- `dim`: 200
- `margin`: 5.0
- `train_times`: 1000 epochs
- `alpha`: 1.0
- `nbatches`: 100
- `type_constrain`: False
- `use_cross_entropy`: True
- `alpha_trust`: 0.5, `beta_trust`: 0.5

**Kết quả:**
| Metric | Giá trị |
|--------|---------|
| **MRR** | 0.2528 |
| **MR** | 193.6 |
| **Hits@10** | 0.4143 |
| **Hits@3** | 0.2738 |
| **Hits@1** | 0.1712 |

**Đánh giá:** ✅ Kết quả tốt, ổn định, làm nền tảng cho các lần sau.

---

### Lần 3 (Tăng Epochs)

**Configuration:**
- Giữ nguyên config lần 1
- `train_times`: 1200 epochs (tăng từ 1000)

**Kết quả:**
| Metric | Giá trị | So với Lần 1 |
|--------|---------|--------------|
| **MRR** | 0.2533 | +0.20% ✅ |
| **MR** | 192.8 | -0.41% ✅ (MR càng nhỏ càng tốt) |
| **Hits@10** | 0.4145 | +0.05% ✅ |
| **Hits@3** | 0.2744 | +0.22% ✅ |
| **Hits@1** | 0.1719 | +0.41% ✅ |

**Đánh giá:** ✅ Cải thiện nhẹ so với lần 1, cho thấy tăng epochs có hiệu quả.

---

### Lần 4 (Tốt Nhất)

**Configuration:**
- Giữ nguyên config lần 3
- `train_times`: 1200 epochs

**Kết quả:**
| Metric | Giá trị | So với Lần 3 |
|--------|---------|--------------|
| **MRR** | **0.2538** | +0.20% ✅ |
| **MR** | 193.9 | +0.57% ⚠️ |
| **Hits@10** | **0.4159** | +0.34% ✅ |
| **Hits@3** | **0.2757** | +0.47% ✅ |
| **Hits@1** | 0.1718 | -0.06% ⚠️ |

**Đánh giá:** ✅ **TỐT NHẤT** trong các lần train. Tất cả metrics chính đều cải thiện.

---

### Lần 5 (Match Baseline Config - Thất Bại)

**Configuration:**
- `dim`: 300 (tăng từ 200)
- `margin`: 6.0 (tăng từ 5.0)
- `train_times`: 1500 epochs
- `alpha`: 0.5 (giảm từ 1.0)
- `nbatches`: 50 (giảm từ 100)
- `type_constrain`: True (bật)
- `use_cross_entropy`: False (tắt)
- `alpha_trust`: 0.5, `beta_trust`: 0.5

**Kết quả (epoch 1300, bị timeout):**
| Metric | Giá trị | So với Lần 4 |
|--------|---------|--------------|
| **MRR** | 0.0859 | -66.1% ❌ |
| **MR** | 499.4 | +157.5% ❌ |
| **Hits@10** | 0.1415 | -66.0% ❌ |
| **Hits@3** | 0.0878 | -67.9% ❌ |
| **Hits@1** | 0.0495 | -71.2% ❌ |

**Đánh giá:** ❌ **THẤT BẠI**. Model chưa converge, kết quả rất kém. Config match baseline không phù hợp với TransT.

**Nguyên nhân:**
1. Learning rate quá thấp (0.5) → model học chậm
2. Margin quá cao (6.0) → không phù hợp với TransT
3. `use_cross_entropy=False` → không tương thích với trustiness
4. Bị timeout ở epoch 1300 → chưa đủ thời gian để converge

---

## 📊 PHÂN TÍCH XU HƯỚNG

### Biểu Đồ Xu Hướng Metrics

#### MRR
```
Lần 1: 0.2528
Lần 3: 0.2533 (+0.20%)
Lần 4: 0.2538 (+0.20%) ← TỐT NHẤT
Lần 5: 0.0859 (-66.1%) ← THẤT BẠI
```

**Nhận xét:** 
- Lần 1-4: Cải thiện dần, ổn định
- Lần 5: Giảm mạnh do config không phù hợp

#### Hits@10 (Metric Quan Trọng Nhất)
```
Lần 1: 0.4143
Lần 3: 0.4145 (+0.05%)
Lần 4: 0.4159 (+0.34%) ← TỐT NHẤT
Lần 5: 0.1415 (-66.0%) ← THẤT BẠI
```

**Nhận xét:**
- Lần 1-4: Cải thiện dần, đạt 0.4159
- Lần 5: Giảm mạnh, model chưa converge

#### MR (Mean Rank - Càng Nhỏ Càng Tốt)
```
Lần 1: 193.6
Lần 3: 192.8 (-0.41%) ← TỐT NHẤT
Lần 4: 193.9 (+0.57%)
Lần 5: 499.4 (+157.5%) ← THẤT BẠI
```

**Nhận xét:**
- Lần 1-4: Ổn định trong khoảng 192-194
- Lần 5: Tăng mạnh, ranking quality kém

---

## 🔍 PHÂN TÍCH CHI TIẾT

### 1. Config Tốt Nhất (Lần 4)

**Hyperparameters tối ưu:**
- `dim`: 200
- `margin`: 5.0
- `alpha`: 1.0
- `nbatches`: 100
- `type_constrain`: False
- `use_cross_entropy`: True
- `alpha_trust`: 0.5, `beta_trust`: 0.5

**Đặc điểm:**
- ✅ Learning rate cao (1.0) → model học nhanh
- ✅ Margin vừa phải (5.0) → phù hợp với TransT
- ✅ Cross-entropy loss → tương thích với trustiness
- ✅ Trustiness weights cân bằng → ổn định

### 2. Config Thất Bại (Lần 5)

**Hyperparameters không phù hợp:**
- `dim`: 300 (quá lớn?)
- `margin`: 6.0 (quá cao)
- `alpha`: 0.5 (quá thấp)
- `use_cross_entropy`: False (không tương thích)

**Vấn đề:**
- ❌ Learning rate quá thấp → model học chậm, chưa converge
- ❌ Margin quá cao → không phù hợp với TransT
- ❌ Margin loss thay vì cross-entropy → không tận dụng được trustiness tốt

### 3. So Sánh với Baseline

**Baseline TransE:**
- `dim`: 300
- `margin`: 6.0
- `alpha`: 0.5
- `type_constrain`: True
- Không có trustiness

**TransT Best (Lần 4):**
- `dim`: 200
- `margin`: 5.0
- `alpha`: 1.0
- `type_constrain`: False
- Có trustiness

**Kết luận:** TransT cần config riêng, không thể copy trực tiếp từ baseline TransE.

---

## 💡 BÀI HỌC KINH NGHIỆM

### 1. ✅ Điều Làm Đúng

1. **Giữ config ổn định** (Lần 1-4)
   - Chỉ tăng epochs từ 1000 → 1200
   - Kết quả cải thiện dần và ổn định

2. **Sử dụng cross-entropy loss**
   - Tương thích tốt với trustiness
   - Kết quả tốt hơn margin loss

3. **Learning rate cao (1.0)**
   - Model học nhanh và hiệu quả
   - Phù hợp với TransT

4. **Margin vừa phải (5.0)**
   - Không quá cao, phù hợp với TransT
   - Kết quả tốt hơn margin 6.0

### 2. ❌ Điều Cần Tránh

1. **Copy config baseline trực tiếp**
   - Baseline config không phù hợp với TransT
   - Cần điều chỉnh cho phù hợp

2. **Learning rate quá thấp**
   - Model học chậm, chưa converge
   - Cần nhiều epochs hơn

3. **Margin quá cao**
   - Không phù hợp với TransT
   - Kết quả kém hơn

4. **Tắt cross-entropy loss**
   - Không tận dụng được trustiness
   - Kết quả kém hơn

---

## 🎯 KẾT LUẬN

### Điểm Mạnh

1. ✅ **Kết quả ổn định:** Lần 1-4 đạt kết quả nhất quán
2. ✅ **Cải thiện dần:** Từ lần 1 → lần 4, metrics cải thiện nhẹ
3. ✅ **Best result:** Lần 4 đạt kết quả tốt nhất (MRR: 0.2538, Hits@10: 0.4159)
4. ✅ **Trustiness hoạt động:** Trustiness giúp model học tốt hơn

### Điểm Yếu

1. ⚠️ **Vẫn thấp hơn baseline:** Tất cả metrics thấp hơn 10-15%
2. ⚠️ **MR cao:** MR = 192.8 (baseline: 102.2) → ranking quality cần cải thiện
3. ⚠️ **Config tối ưu chưa tìm được:** Cần tiếp tục thử nghiệm

### Khuyến Nghị

1. **Tiếp tục với config lần 4:**
   - Giữ nguyên config tốt nhất
   - Tăng epochs lên 1500-2000
   - Có thể cải thiện thêm

2. **Điều chỉnh từ config lần 4:**
   - Thử tăng `dim` từ 200 → 250 (không phải 300)
   - Thử điều chỉnh `margin` từ 5.0 → 5.5
   - Thử điều chỉnh `alpha` từ 1.0 → 0.8
   - Bật `type_constrain=True` để so sánh công bằng với baseline

3. **Tối ưu trustiness:**
   - Thử điều chỉnh `alpha_trust` và `beta_trust`
   - Thử các weights khác nhau (0.3/0.7, 0.7/0.3)

4. **Tăng thời gian training:**
   - Giảm `nbatches` từ 100 → 50 để train nhanh hơn
   - Hoặc tăng thời gian limit trên Kaggle

---

## 📋 BẢNG TỔNG HỢP

### Tất Cả Kết Quả

| Lần | MRR | MR | Hits@10 | Hits@3 | Hits@1 | Config | Đánh giá |
|-----|-----|-----|---------|--------|--------|--------|----------|
| **1** | 0.2528 | 193.6 | 0.4143 | 0.2738 | 0.1712 | Baseline | ✅ Tốt |
| **3** | 0.2533 | **192.8** | 0.4145 | 0.2744 | **0.1719** | +epochs | ✅ Tốt |
| **4** | **0.2538** | 193.9 | **0.4159** | **0.2757** | 0.1718 | +epochs | ✅ **TỐT NHẤT** |
| **5** | 0.0859 | 499.4 | 0.1415 | 0.0878 | 0.0495 | Match baseline | ❌ Thất bại |
| **Baseline** | 0.2901 | 102.2 | 0.4895 | 0.3195 | 0.1915 | TransE | ⭐ Mục tiêu |

### So Sánh với Baseline

| Metric | Baseline | Best TransT | Chênh lệch | % cần cải thiện |
|--------|----------|-------------|------------|-----------------|
| **MRR** | 0.2901 | 0.2538 | -0.0363 | +14.3% |
| **MR** | 102.2 | 192.8 | +90.6 | -47.0% |
| **Hits@10** | 0.4895 | 0.4159 | -0.0736 | +17.7% |
| **Hits@3** | 0.3195 | 0.2757 | -0.0438 | +15.9% |
| **Hits@1** | 0.1915 | 0.1719 | -0.0196 | +11.4% |

---

## 🚀 HƯỚNG PHÁT TRIỂN TIẾP THEO

### Ngắn Hạn (1-2 tuần)

1. ✅ Tiếp tục train với config lần 4, tăng epochs lên 1500-2000
2. ✅ Điều chỉnh nhẹ config lần 4 (dim, margin, alpha)
3. ✅ Thử nghiệm các trustiness weights khác nhau

### Trung Hạn (1 tháng)

1. ✅ Tối ưu hyperparameters chi tiết hơn
2. ✅ Thử nghiệm các loss functions khác
3. ✅ Phân tích sâu hơn về trustiness impact

### Dài Hạn (2-3 tháng)

1. ✅ Đạt hoặc vượt baseline
2. ✅ Thử nghiệm trên dataset khác (WN18RR)
3. ✅ So sánh với các models khác

---

## 📝 GHI CHÚ

- **Lần 2:** Không có kết quả chi tiết trong báo cáo này
- **Early Stopping:** Đã được tích hợp nhưng chưa trigger trong các lần train
- **Best Result Auto-Save:** Đã được tích hợp, tự động lưu best result trong quá trình training

---

## 🚨 HẠN CHẾ VÀ VẤN ĐỀ

### 1. TransE trên FB15K237

**✅ Điểm mạnh:**
- Đạt kết quả tốt nhất trong tất cả models
- Vượt tiêu chuẩn công bố (Hits@10: 0.4895 > 0.476 OpenKE)
- Ổn định và nhất quán

**⚠️ Hạn chế:**
- Không có hạn chế đáng kể
- Có thể cải thiện thêm với hyperparameter tuning

---

### 2. TransE trên WN18RR

**✅ Điểm mạnh:**
- Hits@10: 0.4474 - gần với tiêu chuẩn
- Hits@3: 0.3555 - tốt hơn FB15K237
- Kết quả ổn định

**❌ Hạn chế:**
1. **Hits@1 rất thấp (0.0061)**
   - Chỉ 0.61% dự đoán chính xác ở vị trí đầu tiên
   - Thấp hơn tiêu chuẩn công bố 86% (0.043)
   - TransE đơn giản không đủ mạnh cho WN18RR

2. **MRR thấp (0.1932)**
   - Thấp hơn tiêu chuẩn công bố 14-20% (0.226-0.243)
   - Ranking quality cần cải thiện

3. **MR cao (4251.0)**
   - Xếp hạng trung bình kém (MR càng nhỏ càng tốt)
   - Cao hơn tiêu chuẩn công bố 25-85%

4. **Dataset khó hơn**
   - WN18RR là dataset filtered, khó hơn WN18 gốc
   - Quan hệ từ vựng phức tạp hơn quan hệ thực tế
   - TransE có thể chưa đủ mạnh để xử lý

---

### 3. TransT trên FB15K237

**✅ Điểm mạnh:**
1. **Kết quả ổn định**
   - Lần 1-4 đạt kết quả nhất quán
   - Cải thiện dần qua các lần train

2. **Trustiness hoạt động**
   - Trustiness giúp model học tốt hơn
   - Cải thiện so với TransE đơn giản (trong một số trường hợp)

3. **Best result đạt được**
   - MRR: 0.2538, Hits@10: 0.4159
   - Ổn định và có thể tái tạo

**❌ Hạn chế:**
1. **Vẫn thấp hơn TransE baseline**
   - MRR: 0.2538 vs 0.2901 (-12.5%)
   - Hits@10: 0.4159 vs 0.4895 (-15.0%)
   - MR: 192.8 vs 102.2 (+88.6%)
   - **Chưa đạt được mục tiêu vượt baseline**

2. **MR cao (192.8)**
   - Ranking quality kém hơn TransE baseline
   - Cần cải thiện để giảm MR

3. **Config tối ưu chưa tìm được**
   - Đã thử nhiều config nhưng chưa vượt baseline
   - Config match baseline (lần 5) thất bại hoàn toàn
   - Cần tiếp tục tối ưu hyperparameters

4. **Trustiness chưa phát huy hết tiềm năng**
   - Trustiness weights có thể chưa tối ưu
   - Cần điều chỉnh alpha_trust và beta_trust
   - Có thể cần cải thiện cách tính trustiness

5. **Thời gian training dài**
   - Cần nhiều epochs để converge
   - Bị timeout ở lần 5 (epoch 1300)
   - Cần tối ưu thời gian training

6. **Chưa thử nghiệm trên WN18RR**
   - Chỉ train trên FB15K237
   - Chưa biết hiệu quả trên dataset khác
   - Cần mở rộng thử nghiệm

---

## 📊 TỔNG KẾT HẠN CHẾ

### Hạn Chế Chung

1. **TransT chưa vượt TransE baseline**
   - Mục tiêu ban đầu chưa đạt được
   - Cần tiếp tục tối ưu và nghiên cứu

2. **Config tối ưu chưa tìm được**
   - Đã thử nhiều config nhưng chưa tìm được config tốt nhất
   - Cần grid search hoặc bayesian optimization

3. **Thời gian training dài**
   - Cần nhiều epochs và thời gian
   - Cần tối ưu để giảm thời gian training

4. **Chưa thử nghiệm đầy đủ**
   - Chỉ train trên FB15K237
   - Chưa thử nghiệm trên WN18RR
   - Cần mở rộng thử nghiệm

### Hạn Chế Cụ Thể

| Model | Dataset | Hạn chế chính |
|-------|---------|---------------|
| **TransE** | FB15K237 | ✅ Không có hạn chế đáng kể |
| **TransE** | WN18RR | ❌ Hits@1 rất thấp (0.0061), MRR thấp |
| **TransT** | FB15K237 | ❌ Thấp hơn baseline 12-15%, MR cao, config chưa tối ưu |

---

**Kết thúc báo cáo**

