# Baseline TransE Model

Thư mục này chứa **baseline TransE model** đã được tối ưu trên FB15K237, được dùng làm **tiêu chuẩn so sánh** với các mô hình cải tiến.

## 📊 Baseline Performance

Kết quả trên **FB15K237**:

| Metric | Value |
|--------|-------|
| **MRR** | 0.2901 |
| **MR** | 102.2 |
| **Hits@10** | 0.4895 |
| **Hits@3** | 0.3195 |
| **Hits@1** | 0.1915 |

### So sánh với tiêu chuẩn

| Metric | Baseline (This) | OpenKE Paper | Paper Original |
|--------|-----------------|--------------|----------------|
| **Hits@10** | **0.4895** ✅ | 0.476 | 0.486 |

**✅ Baseline này tốt hơn tiêu chuẩn của tác giả!**

## ⚙️ Baseline Hyperparameters

**⚠️ KHÔNG THAY ĐỔI** các hyperparameters này để giữ làm tiêu chuẩn:

```python
BASELINE_CONFIG = {
    'dim': 300,              # Embedding dimension
    'margin': 6.0,           # Margin for margin loss
    'train_times': 1500,     # Number of epochs
    'alpha': 0.5,            # Learning rate
    'nbatches': 50,          # Number of batches per epoch
    'neg_ent': 10,           # Negative samples per positive
    'threads': 4,            # Number of threads
    'p_norm': 1,             # L1 distance
    'norm_flag': True,       # Normalize embeddings
    'type_constrain': True,  # Use type constraint in evaluation
}
```

## 🚀 Train Baseline

### Trên Kaggle:

Copy code từ `baseline/train_baseline_transe.py` vào Kaggle Notebook và chạy.

### Local (nếu cần):

```bash
python baseline/train_baseline_transe.py
```

## 📝 Sử dụng Baseline

### 1. Train Baseline để có checkpoint

Sử dụng script `train_baseline_transe.py` để train và lưu checkpoint.

### 2. So sánh với mô hình mới

Khi train mô hình cải tiến mới, so sánh kết quả với baseline này:

```python
# Baseline results
baseline_results = {
    'MRR': 0.2901,
    'MR': 102.2,
    'Hits@10': 0.4895,
    'Hits@3': 0.3195,
    'Hits@1': 0.1915
}

# New model results
new_model_results = {
    'MRR': ...,
    'MR': ...,
    'Hits@10': ...,
    'Hits@3': ...,
    'Hits@1': ...
}

# Compare
improvement = (new_model_results['Hits@10'] - baseline_results['Hits@10']) / baseline_results['Hits@10'] * 100
print(f"Improvement: {improvement:.2f}%")
```

## 🔍 Kiểm tra Baseline

Để đảm bảo baseline hoạt động đúng:

1. Train với exact hyperparameters trên
2. Kết quả phải gần với:
   - Hits@10: ~0.4895
   - MRR: ~0.2901
   - MR: ~102.2

Nếu khác nhiều, kiểm tra lại:
- Dataset (FB15K237)
- Hyperparameters
- Evaluation settings (type_constrain=True)

## 📦 Checkpoint

Checkpoint được lưu tại:
- Kaggle: `baseline/checkpoints/baseline_transe.ckpt`
- Local: `./baseline/checkpoints/baseline_transe.ckpt`

## 💡 Lưu ý

1. **Đừng thay đổi hyperparameters** của baseline
2. **Luôn dùng cùng dataset** (FB15K237) để so sánh công bằng
3. **Giữ lại checkpoint** để có thể test lại sau
4. **Document kết quả** của các mô hình mới so với baseline

## 🎯 Mục đích

Baseline này được dùng để:
- ✅ So sánh với các mô hình cải tiến
- ✅ Đánh giá mức độ cải thiện
- ✅ Đảm bảo các thay đổi thực sự tốt hơn
- ✅ Có điểm chuẩn rõ ràng cho nghiên cứu
