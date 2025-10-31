# 🚀 Hướng dẫn chi tiết train OpenKE trên Kaggle

## 📋 Mục lục
1. [Chuẩn bị tài khoản](#1-chuẩn-bị-tài-khoản)
2. [Tạo Notebook mới](#2-tạo-notebook-mới)
3. [Setup môi trường](#3-setup-môi-trường)
4. [Hiểu về Dataset Format](#hiểu-về-dataset-format) ⭐
5. [Upload code OpenKE](#4-upload-code-openke)
6. [Train model](#5-train-model)
7. [Download kết quả](#6-download-kết-quả)

---

## 1. Chuẩn bị tài khoản

### Bước 1.1: Đăng nhập/Đăng ký Kaggle
1. Vào https://www.kaggle.com
2. Click "Sign In" nếu đã có tài khoản
3. Hoặc click "Register" để tạo tài khoản mới (miễn phí)
4. Verify email nếu cần

### Bước 1.2: Verify tài khoản để dùng GPU
1. Vào Profile → Account → Phone number
2. Verify số điện thoại (cần để dùng GPU free)
3. Đợi 1-2 phút để được verify

---

## 2. Tạo Notebook mới

### Bước 2.1: Tạo Notebook
1. Click **"Code"** ở menu trên cùng
2. Chọn **"New Notebook"**
3. Đặt tên: `OpenKE_TransE_Training`

### Bước 2.2: Bật GPU ⚡ (QUAN TRỌNG!)
1. Ở góc trên bên phải, click **"Settings"** (bánh răng ⚙️)
2. Scroll xuống phần **"Environment"**
3. Chọn **"GPU"** trong **"Accelerator"**
4. Click **"Save"** (phải đợi 1-2 phút để GPU được allocate)

> ⚠️ **Note**: GPU free 30 giờ/tuần. Kaggle sẽ disconnect sau 9 giờ, nhưng code sẽ tiếp tục chạy

---

## 3. Setup môi trường

### Cell 1: Clone OpenKE repository
Copy và paste code này vào cell đầu tiên của notebook:

```python
# Clone OpenKE repository
import os
!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && git checkout OpenKE-PyTorch

print("✅ OpenKE cloned successfully!")
```

### Cell 2: Copy files cần thiết
```python
import shutil

# Copy openke module vào working directory
shutil.copytree('OpenKE/openke', 'openke', dirs_exist_ok=True)

# Copy dataset
import subprocess
subprocess.run(['mkdir', '-p', 'benchmarks'])
subprocess.run(['cp', '-r', 'OpenKE/benchmarks/FB15K237', 'benchmarks/'])

print("✅ Files copied!")
print(f"✅ OpenKE path: {os.getcwd()}/openke")
print(f"✅ Dataset path: {os.getcwd()}/benchmarks/FB15K237")
```

### Cell 3: Test import
```python
import sys
import os

# Add openke to path
sys.path.insert(0, os.getcwd())

try:
    import openke
    print("✅ OpenKE imported successfully!")
    
    # Test GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
    else:
        print("⚠️ No GPU available")
        
except Exception as e:
    print(f"❌ Error: {e}")
```

Nhấn **"Run All"** để chạy cả 3 cells. Nếu thấy ✅ thì setup thành công!

---

## 🔍 Hiểu về Dataset Format

### Tại sao dataset là các con số?

**Knowledge Graph** được biểu diễn dưới dạng **Triplets** (bộ 3):
```
(head_entity, relation, tail_entity)
```

**Ví dụ thực tế:**
```
("Barack Obama", "isPresidentOf", "United States")
("Paris", "locatedIn", "France") 
("Python", "isA", "Programming Language")
```

### 📁 Cấu trúc file dataset

Mỗi dataset có format **CHUẨN** gồm các file:

#### 1️⃣ `entity2id.txt` - Ánh xạ Entity → ID số
```
14541                          ← Tổng số entities
/m/027rn    0                 ← Freebase MID → OpenKE ID 0
/m/06cx9    1                 ← Freebase MID → OpenKE ID 1
...
```

**Giải thích format `/m/XXXXX`:**
- `/m/` = Freebase MID (Machine Identifier) prefix
- `027rn` = ID duy nhất của entity trong Freebase
- `0` = ID mà OpenKE gán để training (integer đơn giản)

**Ví dụ thực tế:**
```
"/m/02mjmr" → Thực thể "Barack Obama" trên Freebase
"/m/014lc_" → Thực thể "Paris" trên Freebase
"/m/06cx9"  → Một thực thể khác trong Freebase
```

> 💡 **Lưu ý**: `/m/` là chuẩn của Google Freebase (không còn active, đã được Google đóng lại năm 2015). Data này được extract từ Freebase trước khi đóng.

#### 2️⃣ `relation2id.txt` - Ánh xạ Relation → ID số  
```
237                          ← Tổng số relations
/location/country/...    0   ← Tên relation → ID 0
/tv/tv_program/...       1   ← Tên relation → ID 1
...
```

#### 3️⃣ `train2id.txt` - Training data dạng số
```
272115                       ← Số lượng triplets
0 1 0                        ← Head=0, Relation=1, Tail=0
2 3 1                        ← Head=2, Relation=3, Tail=1
...
```
**Format**: `head_id relation_id tail_id`

### 🌍 Các dataset có sẵn trong OpenKE

| Dataset | Source | Entities | Relations | Triplets | Mô tả |
|---------|--------|----------|-----------|----------|-------|
| **FB15K237** | Freebase | 14,541 | 237 | 272,115 | Chuẩn ngành (bỏ thuộc tính ngược) |
| **FB15K** | Freebase | 14,951 | 1,345 | 592,213 | Đầy đủ hơn |
| **WN18RR** | WordNet | 40,943 | 11 | 93,003 | Dữ liệu từ điển tiếng Anh |
| **WN18** | WordNet | 40,943 | 18 | 151,442 | Bản đầy đủ của WN18RR |
| **YAGO3-10** | YAGO | 123,182 | 37 | 1,079,040 | Large-scale dataset |

### 🎯 Tại sao chọn **FB15K237**?

✅ **Lý do FB15K237 phổ biến:**
- Không có leak (loại bỏ inverse relations) → đánh giá công bằng hơn
- Kích thước vừa phải → train nhanh hơn
- Benchmarks phổ biến → so sánh dễ dàng
- Challenges hơn → evaluate đúng khả năng model

> 💡 **Lưu ý**: File dạng số là **chuẩn ngành**, không phải riêng OpenKE. Tất cả framework KGE đều dùng format này (OpenKE, PyKEEN, DGL-KE, v.v.)

### 🔄 Muốn dùng dataset khác?

OpenKE có sẵn nhiều datasets. Chỉ cần thay đổi 1 dòng:

```python
# Thay vì FB15K237
DATASET_PATH = "./benchmarks/WN18RR/"      # WordNet - từ điển
DATASET_PATH = "./benchmarks/FB15K/"       # Freebase đầy đủ
DATASET_PATH = "./benchmarks/YAGO3-10/"    # Large-scale
```

**So sánh datasets:**

| Dataset | Khó | Dễ train | Dùng khi |
|---------|-----|----------|----------|
| FB15K237 | ⭐⭐⭐⭐⭐ | Nhanh (~3h) | Benchmarks nghiên cứu |
| WN18RR | ⭐⭐⭐ | Nhanh (~2h) | Test nhanh |
| FB15K | ⭐⭐ | Trung bình (~5h) | Reproduce paper cũ |
| YAGO3-10 | ⭐⭐⭐ | Rất chậm (~20h) | Large-scale experiments |

> 💡 **Tips**: Bắt đầu với WN18RR nếu muốn test nhanh hơn (nhỏ hơn FB15K237)

---

## 4. Upload code OpenKE

### Bước 4.1: Clone repo và copy files
(Tiếp tục với các Cell 1, 2, 3 đã thực hiện ở trên)

---

## 5. Train Model

### ⚠️ QUAN TRỌNG: Cách chạy cells

**Từ Cell 4 trở đi, bạn nên:**

1. **Chạy TỪNG CELL MỘT** (khuyến nghị):
   - Click vào cell 4 → Nhấn **"Run"** hoặc **Shift + Enter**
   - Đợi nó chạy xong và thấy ✅
   - Click vào cell 5 → Run
   - Tiếp tục cell 6, 7, 8...

   > ✅ **Ưu điểm**: Dễ debug, thấy được output từng bước, biết cell nào bị lỗi

2. **Hoặc chạy TẤT CẢ** (chỉ dùng khi chắc chắn):
   - Nhấn **"Run All Below"** (hoặc Ctrl + Shift + Enter)
   - Tất cả cells sẽ chạy liên tiếp

   > ⚠️ **Lưu ý**: Cell 7 (training) sẽ chạy ngay → bạn sẽ không kịp xem logs của cell 4-6

**Khuyến nghị:** Chạy từng cell một để theo dõi progress tốt hơn! 🎯

### ⚠️ KHÔNG thể chạy song song

**Câu hỏi thường gặp:** Có thể vừa train vừa test không?

**Trả lời:** ❌ **KHÔNG**

**Lý do:**
- Cell Test cần **checkpoint đã train xong** mới test được
- Model phải train hoàn tất trước khi evaluate
- Jupyter Notebook chạy **tuần tự**, không phải song song
- Test trước khi train → sẽ bị lỗi "checkpoint not found"

**Thứ tự ĐÚNG:**
```
1. Train (3-5 giờ) → Wait until complete ✅
2. THEN Test → Chạy sau khi train xong
```

---

### 📝 Quy trình chạy từng Cell (Chạy lần lượt):

```
Cell 1, 2, 3 → Setup ✅ (đã chạy "Run All")
    ↓
Cell 4 → Import thư viện → Click Run → Đợi ✅
    ↓
Cell 5 → Setup DataLoader → Click Run → Đợi ✅
    ↓
Cell 6 → Define Model → Click Run → Đợi ✅
    ↓
Cell 7 → TRAIN MODEL 🚂 → Click Run → Đợi 3-5 giờ ⏱️
    ↓
Cell 8 → Test Model → Click Run → Đợi ✅
```

---

### Cell 4: Import thư viện
```python
import sys
sys.path.insert(0, '.')

from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TestDataLoader
from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader

print("✅ All imports successful!")
```

**→ Click cell này → Nhấn "Run" (Shift + Enter) → Đợi kết quả ✅**

### Cell 5: Setup DataLoader
```python
# Path to dataset (⚠️ QUAN TRỌNG: Cần dấu / ở cuối!)
DATASET_PATH = "./benchmarks/FB15K237/"
CHECKPOINT_PATH = "./checkpoint"

# Create checkpoint directory
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Training data loader (sử dụng PyTorchLoader cho GPU)
train_dataloader = PyTorchTrainDataLoader(
    in_path = DATASET_PATH, 
    nbatches = 100,
    threads = 4,  # Linux support multiprocessing tốt
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = 5,
    neg_rel = 0
)

print(f"✅ Training dataloader ready!")
print(f"   - Entities: {train_dataloader.get_ent_tot()}")
print(f"   - Relations: {train_dataloader.get_rel_tot()}")
print(f"   - Batch size: {train_dataloader.get_batch_size()}")

# Test data loader
test_dataloader = TestDataLoader(DATASET_PATH, "link")
print("✅ Test dataloader ready!")
```

**→ Click cell này → Nhấn "Run" → Đợi kết quả ✅**

### Cell 6: Define Model
```python
# Define TransE model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 200,  # Embedding dimension
    p_norm = 1,  # L1 distance
    norm_flag = True  # Normalize embeddings
)

print("✅ Model defined!")

# Define loss function (Negative Sampling)
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = 5.0),
    batch_size = train_dataloader.get_batch_size()
)

print("✅ Loss function configured!")
```

**→ Click cell này → Nhấn "Run" → Đợi kết quả ✅**

---

### 📚 Giải thích Epoch là gì?

**Epoch** = **1 lần duyệt toàn bộ dữ liệu training**

- **1 epoch**: Model đã học toàn bộ training data 1 lần
- **1000 epochs**: Model sẽ học training data 1000 lần
- Mỗi lần học, model tự điều chỉnh tham số để phù hợp hơn với dữ liệu

**Ví dụ dễ hiểu:**
```
Bạn có 100 bài toán 
→ Epoch 1: Làm 100 bài lần đầu (có thể sai nhiều)
→ Epoch 2: Làm 100 bài lần 2 (học từ lỗi trước)
→ ...
→ Epoch 1000: Làm 100 bài lần 1000 (đã học kỹ, làm đúng nhiều)
```

**Tại sao cần nhiều epochs?**
- ✅ Model cần thời gian để học patterns
- ✅ Loss giảm dần qua mỗi epoch → model học tốt hơn
- ✅ 1000 epochs là số phổ biến cho knowledge graph embedding
- ⚠️ Quá nhiều epochs → overfitting (học thuộc lòng)
- ⚠️ Quá ít epochs → underfitting (chưa học đủ)

### 🤔 1000 Epoch có nhiều không? So sánh với các lĩnh vực

**Knowledge Graph Embedding:**
- 📊 **Số epoch chuẩn**: 500-2000 epochs
- ✅ **1000 epochs** là CHUẨN, KHÔNG nhiều
- ⏱️ Thời gian: 3-5 giờ (với GPU)
- 📈 Lý do: Embedding space phức tạp, cần nhiều iteration

**So sánh với lĩnh vực khác:**

| Loại Task | Số Epoch Chuẩn | Thời gian | Lý do |
|-----------|----------------|----------|-------|
| **Image Classification** | 10-50 | 15 phút - 2 giờ | Data dễ học hơn |
| **Text Classification** | 5-20 | 10 phút - 1 giờ | Simple features |
| **NLP (BERT fine-tune)** | 2-10 | 30 phút - 3 giờ | Pre-trained model |
| **Knowledge Graph** | **500-2000** | **2-6 giờ** | ✅ Embedding complex |
| **GAN Training** | 50-200 | 2-8 giờ | Adversarial training |

**Tại sao Knowledge Graph cần NHIỀU epochs hơn?**

1. **Embedding space lớn**:
   - FB15K237: 14,541 entities → 200D vectors
   - Cần học mối quan hệ giữa tất cả entities

2. **Negative sampling**:
   - Mỗi triplet cần sample nhiều negatives
   - Phải phân biệt đúng/sai trong không gian lớn

3. **Pattern phức tạp**:
   - 1-1, 1-N, N-1, N-N relations
   - Cần nhiều iteration để capture patterns

4. **Loss landscape**:
   - Loss giảm từ 15 → 0.4
   - Cần nhiều steps để converge

**Khuyến nghị số epochs:**

```
Test nhanh:   100-200 epochs   → 20-40 phút
Production:   500 epochs       → 2-3 giờ  
Paper quality: 1000 epochs    → 3-5 giờ ✅ (chuẩn)
Full research: 2000+ epochs    → 6-10 giờ (overkill)
```

> 💡 **Kết luận**: 1000 epochs là CHUẨN cho Knowledge Graph, không nhiều!

---

### Cell 7: TRAIN MODEL 🚂
```python
print("🚀 Starting training with GPU...")
print("="*50)

# Trainer với GPU
trainer = Trainer(
    model = model, 
    data_loader = train_dataloader, 
    train_times = 1000,  # 1000 epochs - Model sẽ học training data 1000 lần
    alpha = 1.0,  # Learning rate
    use_gpu = True  # ⚡ Sử dụng GPU!
)

# Bắt đầu training
trainer.run()

# Save checkpoint
transe.save_checkpoint(f'{CHECKPOINT_PATH}/transe.ckpt')
print("="*50)
print("✅ Training complete! Checkpoint saved.")
```

**→ Click cell này → Nhấn "Run" → Đợi 3-5 giờ ⏱️**

> ⏱️ **Thời gian**: Với GPU, 1000 epochs sẽ mất khoảng **3-5 giờ**  
> 💡 **Tip**: Có thể leave browser, code vẫn chạy trên server

> 💡 **Muốn test nhanh?** Giảm số epochs xuống 100-200 để test trong 20-40 phút:
> ```python
> train_times = 100,  # Thay vì 1000
> ```

### 📊 Loss sẽ thay đổi như thế nào?

Trong quá trình training, bạn sẽ thấy log như:
```
Epoch 1: Loss = 5.234
Epoch 50: Loss = 2.156
Epoch 100: Loss = 1.234
...
Epoch 1000: Loss = 0.456
```

✅ **Loss giảm** = Model đang học tốt  
⚠️ **Loss tăng/không đổi** = Có thể cần điều chỉnh learning rate

### ⚠️ QUAN TRỌNG: Training KHÔNG có Accuracy!

**Câu hỏi thường gặp:** Tại sao không thấy Accuracy trong log?

**Trả lời:** ⚠️ **Đây là BÌNH THƯỜNG!**

**Training chỉ log LOSS:**
```
Epoch 1:  Loss = 15.234  ← Đây là LOSS, không phải accuracy!
Epoch 50: Loss = 8.156
Epoch 100: Loss = 5.234
```

**Tại sao không có accuracy?**
- Training dùng **LOSS** để học (loss giảm = tốt hơn)
- **Accuracy** chỉ có khi **TEST** (Cell 8)
- Đây là chuẩn của Knowledge Graph Embedding, không phải bug!

**Khi nào thấy Accuracy?**
```
Train xong → Chạy Cell 8 (TEST) → SẼ thấy:
             ✅ MRR: 0.456
             ✅ Hits@10: 0.567
             ✅ Accuracy metrics
```

> 💡 **Tóm tắt:** Training = Loss | Testing = Accuracy/Metrics

### 🖥️ Monitor GPU Usage

Khi training, bạn sẽ thấy GPU monitor bên phải:

**✅ GPU hoạt động tốt khi:**
- GPU Status: **"On"** hoặc **"P100 On"**
- GPU Memory > 0 (ví dụ: 427 MiB / 16 GiB)
- Training đang chạy: "🚀 Starting training with GPU..."

**⚠️ GPU usage 0.00% có bình thường không?**
- **CÓ** - Rất bình thường!
- GPU làm việc theo **batches**, không phải liên tục
- Trong lúc load data, GPU có thể idle
- Miễn là GPU Memory > 0 và training vẫn chạy là OK

**Người dùng báo cáo:**
```
Epoch 32 | loss: 15.043990: 3% | 33/1000 [09:56<4:49:16, 17.95s/it]
```

✅ Đây là **CHÍNH XÁC** - đang train epoch 33/1000 trên GPU!  
⏱️ Đã chạy 9 phút 56 giây  
📊 Loss = 15.04 (sẽ giảm dần)  
🎯 ETA còn ~4 giờ 49 phút

---

## 5. Test Model

### Cell 8: Evaluate Model
```python
print("🧪 Testing model...")
print("="*50)

# Load checkpoint
transe.load_checkpoint(f'{CHECKPOINT_PATH}/transe.ckpt')

# Test với GPU
tester = Tester(
    model = transe, 
    data_loader = test_dataloader, 
    use_gpu = True
)

# Run link prediction
tester.run_link_prediction(type_constrain = False)

print("="*50)
print("✅ Testing complete!")
```

**→ Click cell này → Nhấn "Run" → Đợi kết quả (khoảng 5-10 phút)**

Bạn sẽ thấy kết quả như:
```
MR: 123.4 (Mean Rank)
MRR: 0.456
Hits@10: 0.567
Hits@3: 0.489
Hits@1: 0.345
```

---

## 6. Download kết quả

### Cách 1: Download từ Kaggle
1. Scroll xuống phần **"Output"** bên dưới
2. Click vào `checkpoint/` folder
3. Download file `transe.ckpt`

### Cách 2: Save vào Kaggle Output
```python
# File sẽ tự động save vào /kaggle/working/checkpoint/
# Bạn có thể download về máy sau khi chạy xong
```

---

## 📊 Tracking Progress

Kaggle sẽ tự động log:
- Loss mỗi epoch
- Time per epoch
- ETA (estimated time to completion)

Bạn có thể theo dõi real-time trong output cell.

---

## 🔧 Troubleshooting

### Lỗi: "CUDA out of memory"
```python
# Giảm batch size
train_dataloader = PyTorchTrainDataLoader(
    nbatches = 200,  # Tăng số batches = giảm batch size
    # ... rest of config
)
```

### Lỗi: "Session expired"
- Kaggle disconnect sau 9 giờ
- Code vẫn chạy, check kết quả sau
- Hoặc chia nhỏ training: train 500 epochs → save → tiếp tục

### Lỗi: "No GPU available"
- Kiểm tra Settings → Accelerator = GPU
- Đợi 1-2 phút để GPU được allocate

### Lỗi: "FileNotFoundError: ./benchmarks/FB15K237entity2id.txt"
```python
# ❌ SAI: Thiếu dấu / ở cuối
DATASET_PATH = "./benchmarks/FB15K237"

# ✅ ĐÚNG: Cần dấu / ở cuối
DATASET_PATH = "./benchmarks/FB15K237/"
```
**Nguyên nhân**: PyTorchTrainDataLoader sẽ nối trực tiếp tên file vào path, nên cần dấu `/` ở cuối.

---

## 📈 Expected Results

Với FB15K237 dataset:
- **Training time**: 3-5 giờ (GPU)
- **MRR**: ~0.35-0.40
- **Hits@10**: ~0.45-0.50
- **Checkpoint size**: ~50-100MB

---

## 💡 Tips

1. **Save frequently**: Nếu training lâu, save checkpoint mỗi 200 epochs
2. **Monitor loss**: Loss giảm dần = model đang học tốt
3. **Download results**: File save trong `/kaggle/working/` sẽ persist sau khi disconnect

---

## ❓ FAQ

### **`/m/027rn    0` có nghĩa là gì?**

Đây là **Freebase MID** (Machine Identifier) - mã định danh machine-readable của Google Freebase:

**Chi tiết:**
```
/m/027rn    → Freebase MID (từ Google Freebase database)
0           → OpenKE ID (số đơn giản để training)
```

**Giải thích:**
- `/m/` = Prefix của Freebase MID format
- `027rn` = ID duy nhất được Freebase gán (base62 encoded)
- `0` = ID số đơn giản mà OpenKE sử dụng để training

**Mapping thực tế:**
```
"/m/02mjmr" → "Barack Obama" trên Freebase
"/m/014lc_" → "Paris, France" trên Freebase  
"/m/027rn"  → Thực thể khác trong Freebase
```

> 📝 **Lịch sử**: Freebase là knowledge graph được Google sở hữu (2007-2015). Sau khi Google đóng (2015), data được extract và dùng cho research. Format `/m/` là legacy từ Freebase.

**Tại sao không dùng tên thật?**
- Tên entity có thể thay đổi, trùng lặp, ngôn ngữ khác nhau
- MID là unique, immutable, machine-readable
- Giống như ISBN cho sách - ID duy nhất và chuẩn

### **Tại sao training không có Accuracy?**

❌ **Đây KHÔNG phải lỗi - Đây là BÌNH THƯỜNG!**

**Sự khác biệt giữa LOSS và ACCURACY:**

| | LOSS | ACCURACY |
|---|---|---|
| **Là gì?** | Sai số (error) | Độ chính xác |
| **Train dùng?** | ✅ Có | ❌ Không |
| **Test dùng?** | ❌ Không | ✅ Có |
| **Giá trị** | Số càng thấp càng tốt | Số càng cao càng tốt (0-1) |

**Ví dụ:**
```
Training (Cell 7):
Epoch 1:  Loss = 15.234  ← Chỉ có LOSS!
Epoch 50: Loss = 8.156   ← Loss giảm = tốt
Epoch 1000: Loss = 0.456 ← Loss càng thấp càng tốt

Testing (Cell 8):
MRR = 0.456      ← Accuracy metrics!
Hits@10 = 0.567  ← Accuracy!
Hits@1 = 0.345   ← Accuracy!
```

**Tại sao Training chỉ có Loss?**
- Model học bằng cách **minimize loss** (giảm sai số)
- Loss càng thấp = embedding càng tốt
- Accuracy **không có trong training** vì cần test set
- Đây là chuẩn của **Knowledge Graph Embedding**

> 💡 **Kết luận:** Training = Log Loss | Testing = Log Accuracy/Metrics

---

### 🤔 So sánh với các lĩnh vực khác

**Câu hỏi:** Các lĩnh vực khác (Image Classification, NLP) có hiện Accuracy khi train không?

**Trả lời:**

#### ✅ CÓ hiện Accuracy trong Training:

**Image Classification (Ví dụ: PyTorch):**
```python
# Train
Epoch 1: Train Loss = 1.234 | Train Acc = 0.567 ✅
Epoch 2: Train Loss = 0.987 | Train Acc = 0.678 ✅

# Test  
Epoch 1: Test Acc = 0.589 ✅
```

**NLP Text Classification:**
```python
Epoch 1: Loss = 0.456 | Accuracy = 0.789 ✅
```

**Tabular Data (Scikit-learn):**
```python
model.fit(X_train, y_train)
# Train score: 0.85 ✅
# Test score: 0.82 ✅
```

#### ❌ KHÔNG có Accuracy trong Training:

**Knowledge Graph Embedding (OpenKE, TransE, v.v.):**
```python
# Train
Epoch 1: Loss = 15.234  # ❌ CHỈ có Loss
Epoch 2: Loss = 8.156   # ❌ CHỈ có Loss
...
Epoch 1000: Loss = 0.456 # ❌ CHỈ có Loss

# Test - MỚI có Accuracy
MRR = 0.456 ✅
Hits@10 = 0.567 ✅
```

#### 📊 Tại sao khác nhau?

| Loại Task | Có Accuracy trong Train? | Lý do |
|-----------|--------------------------|-------|
| **Image Classification** | ✅ CÓ | Có label cụ thể → tính accuracy dễ |
| **Text Classification** | ✅ CÓ | Có class cụ thể → tính accuracy |
| **Regression** | ❌ KHÔNG | Dùng MSE/RMSE (tương tự loss) |
| **Knowledge Graph** | ❌ KHÔNG | Không có "đúng/sai" rõ ràng, chỉ có ranking |

**Knowledge Graph đặc biệt:**
- Không có "label" cụ thể như Image/Text
- Chỉ có triplet (head, relation, tail)
- Không đo "accuracy" mà đo **ranking quality**:
  - MRR: Mean Reciprocal Rank
  - Hits@K: Trong top K có đúng không?
- Accuracy chỉ có trong **testing** để đánh giá ranking

> 💡 **Kết luận**: 
> - Image/Text: Train có Accuracy ✅
> - Knowledge Graph: Train KHÔNG có Accuracy ❌ (chỉ có khi Test)

### **Có thể vừa train vừa test không?**

❌ **KHÔNG thể chạy song song** Training và Testing.

**Lý do:**
```
✅ ĐÚNG: Train xong → RỒI mới Test
Train (Cell 7) → Đợi 3-5 giờ → Checkpoint saved → Test (Cell 8)

❌ SAI: Train và Test cùng lúc
Train (Cell 7) + Test (Cell 8) chạy cùng lúc
→ ERROR: Checkpoint not found!
```

**Nguyên nhân kỹ thuật:**
1. **Jupyter Notebook chạy TUẦN TỰ** - Cell sau phải đợi cell trước
2. **Test cần model đã train** - Load checkpoint từ cell 7
3. **Nếu train chưa xong** → Checkpoint chưa có → Test fail

**Quy trình ĐÚNG:**
```
Cell 1-3: Setup ✅
Cell 4-5: Import & Load data ✅  
Cell 6: Define model ✅
Cell 7: TRAIN 🚂 ⏰ (3-5 giờ - KHÔNG chạy cell khác)
         ↓ Đợi hoàn thành
Cell 8: TEST 🧪 ✅ (Chạy SAU KHI train xong)
```

---

## ✅ Checklist

- [ ] Đăng nhập Kaggle
- [ ] Verify phone number
- [ ] Tạo Notebook mới
- [ ] Bật GPU
- [ ] Clone OpenKE repo
- [ ] Copy files
- [ ] Setup import
- [ ] Define model
- [ ] Train model
- [ ] Test model
- [ ] Download checkpoint

**Chúc bạn thành công! 🎉**


