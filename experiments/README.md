# Experiments Directory

Thư mục này chứa các script thực nghiệm train TransE trên các dataset khác nhau.

## 📁 Các Experiment

### 1. WN18RR Dataset

**File**: `train_transe_wn18rr.py`

Train TransE trên dataset WN18RR (WordNet - lexical relations).

#### Hyperparameters (đã điều chỉnh):
- `dim`: 100
- `margin`: 3.0 (tăng từ 1.0)
- `train_times`: 2000 epochs (tăng từ 1000)
- `alpha`: 0.1 (tăng từ 0.01 - learning rate)
- `nbatches`: 50
- `neg_ent`: 5 (tăng từ 1)
- `type_constrain`: False

**Lý do điều chỉnh:** Learning rate 0.01 quá nhỏ khiến model không học được.

#### Đặc điểm WN18RR:
- Dataset nhỏ hơn FB15K237
- Ít entities và relations hơn
- Cần hyperparameters khác với FB15K237

## 🚀 Sử dụng

### Train trên Kaggle:

1. Mở file `train_transe_wn18rr.py`
2. Copy từng cell vào Kaggle Notebook
3. Enable GPU và chạy

### So sánh với Baseline:

Baseline FB15K237:
- Hits@10: 0.4895
- MRR: 0.2901

So sánh kết quả WN18RR với baseline để đánh giá hiệu suất trên dataset khác.

## 📊 Kết quả mong đợi

Theo OpenKE paper, TransE trên WN18RR thường đạt:
- Hits@10: ~0.512 (theo OpenKE implementation)
- Hits@10: ~0.501 (theo paper gốc)

## 💡 Tips

1. **Learning rate**: 0.1 (không quá nhỏ như 0.01)
2. **Dimension**: 100 (nhỏ hơn FB15K237)
3. **Training time**: ~2-3 giờ với 2000 epochs
4. Nếu kết quả vẫn thấp, thử tăng `alpha` lên 0.2 hoặc 0.5

## 📝 Thêm Experiment mới

Để thêm experiment cho dataset khác:

1. Copy template từ `train_transe_wn18rr.py`
2. Thay đổi:
   - Dataset path
   - Hyperparameters phù hợp
   - Checkpoint path
3. Chạy và ghi lại kết quả
