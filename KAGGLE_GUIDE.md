# Hướng dẫn train OpenKE trên Kaggle

## ✅ Lợi ích:
- GPU miễn phí (30h/tuần)
- Tốc độ train nhanh hơn 10-20x
- Không bị lỗi multiprocessing (Linux)
- Setup đơn giản

## 📝 Các bước:

### Bước 1: Chuẩn bị dữ liệu
1. Vào https://www.kaggle.com
2. Tạo Kaggle Notebook mới (nhấn "New Notebook")
3. **Enable GPU** (Settings → Accelerator → GPU)

### Bước 2: Upload dữ liệu
#### Option A: Upload dataset
1. Click "Add data" → "Upload a dataset"
2. Upload thư mục `benchmarks/FB15K237/` (zip lại)
3. Attach dataset vào notebook

#### Option B: Clone từ Git
```python
# Cell đầu tiên của notebook
!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && git checkout OpenKE-PyTorch

# Copy data
!mkdir -p benchmarks
!cp -r OpenKE/benchmarks/FB15K237 benchmarks/
```

### Bước 3: Setup code
Copy các file cần thiết vào working directory:
```python
# Cell setup
import shutil
import os

# Copy openke module
shutil.copytree('OpenKE/openke', 'openke', dirs_exist_ok=True)

# Verify
print("Files ready!")
```

### Bước 4: Train model
Copy code từ `kaggle_train_script.py` và chạy trong notebook.

## ⚡ Ước tính thời gian:
- CPU (local): ~74 giờ cho 1000 epochs
- GPU (Kaggle): ~3-5 giờ cho 1000 epochs
- **Tiết kiệm 70 giờ!**

## 💡 Tips:
- Kaggle sessions chạy tối đa 9 giờ
- Nếu cần train lâu hơn, chia nhỏ epochs
- Sử dụng wandb để track progress khi bị disconnect

## 🎯 Lệnh train trên Kaggle:
```bash
# Trong Kaggle Notebook, đặt GPU enabled
# Copy code từ kaggle_train_script.py và chạy
```



