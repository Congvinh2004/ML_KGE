"""
Train TransE trên WN18RR Dataset

Code này train TransE với hyperparameters được tối ưu cho WN18RR
Dataset: WN18RR (WordNet - lexical relations)

Hyperparameters được tối ưu cho WN18RR:
- dim: 100 (nhỏ hơn FB15K237 vì dataset nhỏ hơn)
- margin: 1.0 (nhỏ hơn FB15K237)
- train_times: 1000
- alpha: 0.01 (learning rate nhỏ hơn)
- nbatches: 50
- neg_ent: 1 (ít negative samples)

"""
# ===========================================
# CELL 1: Clone OpenKE repository
# ===========================================
import os

# Clone OpenKE repository
!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && git checkout OpenKE-PyTorch

print("✅ OpenKE cloned successfully!")

# ============================================
# CELL 2: Copy files and setup
# ============================================
import shutil

# Copy openke module vào working directory
shutil.copytree('OpenKE/openke', 'openke', dirs_exist_ok=True)

# Copy dataset
import subprocess
subprocess.run(['mkdir', '-p', 'benchmarks'])
subprocess.run(['cp', '-r', 'OpenKE/benchmarks/WN18RR', 'benchmarks/'])

print("✅ Files copied!")
print(f"✅ OpenKE path: {os.getcwd()}/openke")
print(f"✅ Dataset path: {os.getcwd()}/benchmarks/WN18RR")

# ============================================
# CELL 3: Import và kiểm tra GPU
# ============================================
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

# ============================================
# CELL 4: Build native library
# ============================================
# Build native library for TestDataLoader (Base.so)
!apt-get -y update && apt-get -y install build-essential
%cd openke
!bash make.sh
%cd ..
import os
print("Base.so exists:", os.path.exists("openke/release/Base.so"))

# ============================================
# CELL 5: Import modules
# ============================================
import sys
sys.path.insert(0, '.')

from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TestDataLoader
from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader

print("✅ All imports successful!")

# ============================================
# CELL 6: Setup và Configuration - WN18RR
# ============================================
# Path to dataset
DATASET_PATH = "./benchmarks/WN18RR/"
CHECKPOINT_PATH = "./experiments/checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# ⭐ WN18RR HYPERPARAMETERS - Được tối ưu cho WN18RR
print("="*60)
print("⚙️  WN18RR CONFIGURATION")
print("="*60)

# ⚠️ CẬP NHẬT: Hyperparameters đã được điều chỉnh để cải thiện kết quả
# Kết quả trước: Hits@10 = 0.0346 (quá thấp)
# Nguyên nhân: Learning rate quá nhỏ (0.01) → model không học được
WN18RR_CONFIG = {
    'dim': 100,              # Embedding dimension
    'margin': 3.0,           # ⬆️ Tăng từ 1.0 lên 3.0 (để tách biệt tốt hơn)
    'train_times': 2000,     # ⬆️ Tăng từ 1000 lên 2000 epochs
    'alpha': 0.1,            # ⬆️ Tăng từ 0.01 lên 0.1 (learning rate)
    'nbatches': 50,          # Number of batches per epoch
    'neg_ent': 5,            # ⬆️ Tăng từ 1 lên 5 (nhiều negative samples hơn)
    'threads': 4,            # Number of threads
    'p_norm': 1,             # L1 distance
    'norm_flag': True,       # Normalize embeddings
    'type_constrain': False, # Use type constraint in evaluation
}

print("WN18RR Hyperparameters (Đã điều chỉnh):")
for key, value in WN18RR_CONFIG.items():
    print(f"   {key}: {value}")

print("\n📊 Thay đổi so với config cũ:")
print("   ⬆️ margin: 1.0 → 3.0")
print("   ⬆️ train_times: 1000 → 2000")
print("   ⬆️ alpha: 0.01 → 0.1 (learning rate)")
print("   ⬆️ neg_ent: 1 → 5")
print("\n💡 Lý do: Learning rate quá nhỏ (0.01) khiến model không học được")
print("   Kết quả trước: Hits@10 = 0.0346 (quá thấp)")
print("   Kỳ vọng: Hits@10 ≈ 0.512")

# ============================================
# CELL 7: Load data
# ============================================
# Training data loader (sử dụng PyTorchLoader cho GPU)
train_dataloader = PyTorchTrainDataLoader(
    in_path = DATASET_PATH, 
    nbatches = WN18RR_CONFIG['nbatches'],
    threads = WN18RR_CONFIG['threads'],  # Linux support multiprocessing tốt
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = WN18RR_CONFIG['neg_ent'],
    neg_rel = 0
)

print(f"✅ Training dataloader ready!")
print(f"   - Entities: {train_dataloader.get_ent_tot()}")
print(f"   - Relations: {train_dataloader.get_rel_tot()}")
print(f"   - Batch size: {train_dataloader.get_batch_size()}")

# Test data loader
test_dataloader = TestDataLoader(DATASET_PATH, "link")
print("✅ Test dataloader ready!")

# ============================================
# CELL 8: Define TransE model
# ============================================
# Define TransE model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = WN18RR_CONFIG['dim'],  # Embedding dimension
    p_norm = WN18RR_CONFIG['p_norm'],  # L1 distance
    norm_flag = WN18RR_CONFIG['norm_flag']  # Normalize embeddings
)

print("✅ Model defined!")

# Define loss function (Negative Sampling)
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = WN18RR_CONFIG['margin']),
    batch_size = train_dataloader.get_batch_size()
)

print("✅ Loss function configured!")

# ============================================
# CELL 9: Train Model
# ============================================
print("🚀 Starting training with GPU...")
print("="*50)

# Trainer với GPU
trainer = Trainer(
    model = model, 
    data_loader = train_dataloader, 
    train_times = WN18RR_CONFIG['train_times'],  # 2000 epochs (đã tăng từ 1000)
    alpha = WN18RR_CONFIG['alpha'],  # Learning rate (0.1 - đã tăng từ 0.01)
    use_gpu = True  # ⚡ Sử dụng GPU!
)

# Bắt đầu training
trainer.run()

# Save checkpoint
checkpoint_path = f'{CHECKPOINT_PATH}/transe_wn18rr.ckpt'
transe.save_checkpoint(checkpoint_path)
print("="*50)
print("✅ Training complete! Checkpoint saved.")
print(f"✅ Checkpoint: {checkpoint_path}")

# ============================================
# CELL 10: Test Model
# ============================================
print("🧪 Testing model...")
print("="*50)

# Load checkpoint
transe.load_checkpoint(checkpoint_path)

# Test với GPU
tester = Tester(
    model = transe, 
    data_loader = test_dataloader, 
    use_gpu = True
)

# Run link prediction
tester.run_link_prediction(type_constrain = WN18RR_CONFIG['type_constrain'])

print("="*50)
print("✅ Testing complete!")

# ============================================
# CELL 11: Final Results
# ============================================
print("\n" + "="*60)
print("📊 WN18RR RESULTS")
print("="*60)

mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
print(f"\nMRR: {mrr:.4f}  MR: {mr:.1f}  Hits@10: {hit10:.4f}  Hits@3: {hit3:.4f}  Hits@1: {hit1:.4f}")

print("\n" + "="*60)
print("✅ Training completed!")
print("="*60)
print(f"\n📥 Checkpoint saved: {checkpoint_path}")
print(f"💡 So sánh với baseline FB15K237 để đánh giá hiệu suất")

# ============================================
# CELL 12: Download Checkpoint (Nhiều cách)
# ============================================
print("\n" + "="*60)
print("📥 DOWNLOAD CHECKPOINT")
print("="*60)

# Cách 1: Tạo link download (có thể bị lỗi với file lớn)
try:
    from IPython.display import FileLink
    print(f"\n🔗 Cách 1: Click link để download:")
    print(FileLink(checkpoint_path))
except:
    print("⚠️ Không thể tạo link download (file có thể quá lớn)")

# Cách 2: Zip checkpoint để download dễ hơn
print(f"\n📦 Cách 2: Zip checkpoint để download...")
import zipfile
import os

zip_path = f'{CHECKPOINT_PATH}/transe_wn18rr.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(checkpoint_path, os.path.basename(checkpoint_path))

print(f"✅ Checkpoint đã được zip: {zip_path}")
print(f"   File size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

try:
    from IPython.display import FileLink
    print(f"\n🔗 Click link để download zip file:")
    print(FileLink(zip_path))
except:
    print("⚠️ Không thể tạo link, thử cách 3")

# Cách 3: Hướng dẫn download thủ công
print(f"\n📋 Cách 3: Download thủ công:")
print(f"   1. Vào Output section (bên trái)")
print(f"   2. Vào: experiments/checkpoints/")
print(f"   3. Right-click vào file 'transe_wn18rr.zip'")
print(f"   4. Chọn 'Download' hoặc 'Save as'")

# Cách 4: Sử dụng Kaggle Dataset (nếu cần lưu lâu dài)
print(f"\n💡 Cách 4: Upload lên Kaggle Dataset (để lưu lâu dài):")
print(f"   1. Vào https://www.kaggle.com/datasets")
print(f"   2. Click 'New Dataset'")
print(f"   3. Upload file {zip_path}")
print(f"   4. Publish dataset để lưu trữ")

print("\n⚠️ IMPORTANT: Download checkpoint before session ends!")
print("Kaggle will delete all files when session expires.")
