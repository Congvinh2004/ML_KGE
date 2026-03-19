# 🛑 Hướng dẫn sử dụng Early Stopping và Test Định Kỳ

## 📋 Tổng quan

Phiên bản cải tiến này thêm 2 tính năng quan trọng:
1. **Test định kỳ**: Test model mỗi 50 epochs để theo dõi convergence
2. **Early Stopping**: Tự động dừng training nếu model không cải thiện trong một khoảng % epochs nhất định

## 🚀 Cách sử dụng

### Bước 1: Copy file helper vào workspace

File `early_stopping_helper.py` cần được copy vào thư mục `research/ttm/experiments/` trong workspace của bạn.

### Bước 2: Sử dụng CELL 10 mới

Thay thế CELL 10 trong file `kaggle_train_ttm_FB15K237_v1_config.py` bằng code từ file:
- `kaggle_train_ttm_FB15K237_v2_with_early_stopping.py`

Hoặc copy code từ CELL 10 trong file v2 vào notebook của bạn.

## ⚙️ Cấu hình Early Stopping

### Cấu hình mặc định:

```python
EARLY_STOPPING_CONFIG = {
    'patience_percent': 0.12,    # 12% tổng epochs (ví dụ: 150 epochs cho 1200 epochs)
    'min_delta': 0.0001,         # Cải thiện tối thiểu để coi là "cải thiện"
    'monitor': 'Hits@10',        # Metric để theo dõi (quan trọng nhất)
    'mode': 'max',               # 'max' cho Hits@10, MRR (càng lớn càng tốt)
    'min_epochs': 500,           # Tối thiểu train 500 epochs trước khi early stop
}
```

### Giải thích các tham số:

1. **`patience_percent`**: 
   - Phần trăm tổng epochs để chờ đợi trước khi early stop
   - Ví dụ: `0.12` = 12% = 150 epochs cho 1200 epochs
   - Khuyến nghị: `0.10-0.15` (10-15%)

2. **`min_delta`**: 
   - Cải thiện tối thiểu để coi là "cải thiện"
   - Ví dụ: `0.0001` = 0.01% cải thiện
   - Khuyến nghị: `0.0001-0.001`

3. **`monitor`**: 
   - Metric để theo dõi
   - Có thể: `'Hits@10'`, `'MRR'`, `'MR'`, `'Hits@3'`, `'Hits@1'`
   - Khuyến nghị: `'Hits@10'` (quan trọng nhất)

4. **`mode`**: 
   - `'max'` cho Hits@10, MRR (càng lớn càng tốt)
   - `'min'` cho MR (càng nhỏ càng tốt)
   - Khuyến nghị: `'max'` cho Hits@10

5. **`min_epochs`**: 
   - Tối thiểu train N epochs trước khi early stop
   - Khuyến nghị: `500-800` epochs

### Cấu hình Test Định Kỳ:

```python
TEST_INTERVAL = 50  # Test mỗi 50 epochs
```

- Test sẽ chạy ở các epoch: 50, 100, 150, 200, ..., và epoch cuối cùng
- Có thể điều chỉnh: `25`, `50`, `100` epochs

## 📊 Ví dụ hoạt động

### Với config 1200 epochs:

```
Epoch 50:  Test → Hits@10 = 0.3800 (best: 0.3800)
Epoch 100: Test → Hits@10 = 0.3950 (best: 0.3950) ✅ Improved
Epoch 150: Test → Hits@10 = 0.4100 (best: 0.4100) ✅ Improved
Epoch 200: Test → Hits@10 = 0.4150 (best: 0.4150) ✅ Improved
Epoch 250: Test → Hits@10 = 0.4170 (best: 0.4170) ✅ Improved
Epoch 300: Test → Hits@10 = 0.4175 (best: 0.4175) ✅ Improved
Epoch 350: Test → Hits@10 = 0.4176 (best: 0.4176) ✅ Improved
Epoch 400: Test → Hits@10 = 0.4175 (no improvement, counter: 1/144)
Epoch 450: Test → Hits@10 = 0.4174 (no improvement, counter: 2/144)
...
Epoch 1100: Test → Hits@10 = 0.4176 (no improvement, counter: 144/144)
🛑 Early stopping triggered!
   Best Hits@10: 0.4176 at epoch 350
```

### Với config 1500 epochs:

```
Epoch 50:  Test → Hits@10 = 0.3800 (best: 0.3800)
...
Epoch 1300: Test → Hits@10 = 0.4176 (no improvement, counter: 180/180)
🛑 Early stopping triggered!
   Best Hits@10: 0.4176 at epoch 1200
```

## 📁 Files được tạo

Sau khi training, các files sau sẽ được tạo:

1. **Checkpoints:**
   - `transt_fb15k237_l{TRAIN_RUN}.ckpt` - Checkpoint cuối cùng
   - `transt_fb15k237_l{TRAIN_RUN}_best.ckpt` - **Best checkpoint** (quan trọng nhất!)
   - `transt_fb15k237_l{TRAIN_RUN}_periodic/epoch_XXXXX.ckpt` - Checkpoints định kỳ

2. **Results:**
   - `transt_fb15k237_l{TRAIN_RUN}_results.txt` - Kết quả text
   - `transt_fb15k237_l{TRAIN_RUN}_history.json` - **Lịch sử metrics** (quan trọng!)

3. **CSV:**
   - `transt_fb15k237_all_results.csv` - Tất cả kết quả để so sánh

## 📈 Phân tích kết quả

### File `history.json` chứa:

```json
{
  "patience_percent": 0.12,
  "patience_epochs": 144,
  "monitor": "Hits@10",
  "mode": "max",
  "best_epoch": 350,
  "best_score": 0.4176,
  "history": [
    {
      "epoch": 50,
      "metrics": {
        "MRR": 0.2400,
        "MR": 200.0,
        "Hits@10": 0.3800,
        "Hits@3": 0.2500,
        "Hits@1": 0.1500
      },
      "score": 0.3800
    },
    ...
  ]
}
```

### Cách đọc:

1. **Best epoch**: Epoch tốt nhất (theo metric được monitor)
2. **Best score**: Score tốt nhất
3. **History**: Lịch sử tất cả metrics qua các epochs

## 💡 Khuyến nghị

### 1. Patience Percent:

- **1200 epochs**: `patience_percent = 0.12` → 144 epochs patience
- **1500 epochs**: `patience_percent = 0.12` → 180 epochs patience
- **1000 epochs**: `patience_percent = 0.15` → 150 epochs patience

### 2. Test Interval:

- **Nhanh (ít test)**: `TEST_INTERVAL = 100` epochs
- **Cân bằng**: `TEST_INTERVAL = 50` epochs (khuyến nghị)
- **Chi tiết (nhiều test)**: `TEST_INTERVAL = 25` epochs

### 3. Monitor Metric:

- **Hits@10**: Quan trọng nhất, thường được dùng trong papers
- **MRR**: Tốt cho overall performance
- **MR**: Càng nhỏ càng tốt (dùng `mode='min'`)

## ⚠️ Lưu ý

1. **Best checkpoint**: Luôn dùng `_best.ckpt` để test cuối cùng, không phải checkpoint cuối cùng
2. **Early stopping**: Có thể dừng sớm, tiết kiệm thời gian
3. **History file**: Quan trọng để phân tích convergence
4. **Test định kỳ**: Tốn thời gian, nhưng giúp theo dõi tốt hơn

## 🔧 Troubleshooting

### Lỗi: "Module not found: early_stopping_helper"

**Giải pháp:**
```python
# Thêm vào đầu CELL 10:
import sys
sys.path.insert(0, 'research/ttm/experiments')
```

### Lỗi: "Variables not defined"

**Giải pháp:** Đảm bảo đã chạy các CELL trước đó:
- CELL 6: Config
- CELL 7: Load data
- CELL 8: Setup Trustiness
- CELL 9: Create model

### Early stopping không hoạt động

**Kiểm tra:**
1. `min_epochs` có quá lớn không?
2. `patience_percent` có quá lớn không?
3. `min_delta` có quá nhỏ không?

## 📝 So sánh với phiên bản cũ

| Tính năng | V1 (Cũ) | V2 (Mới) |
|-----------|---------|----------|
| Test | Chỉ test ở cuối | Test định kỳ mỗi 50 epochs |
| Early stopping | Không có | Có (dựa trên % epochs) |
| Best checkpoint | Không có | Tự động lưu |
| History | Không có | Lưu lịch sử metrics |
| Theo dõi convergence | Không | Có |

## ✅ Kết luận

Phiên bản mới giúp:
- ✅ **Tiết kiệm thời gian**: Early stopping khi không cải thiện
- ✅ **Theo dõi tốt hơn**: Test định kỳ để biết model đang học như thế nào
- ✅ **Best model**: Tự động lưu model tốt nhất
- ✅ **Phân tích**: History file để phân tích convergence

**Khuyến nghị: Sử dụng phiên bản mới này cho tất cả các lần train tiếp theo!**
















