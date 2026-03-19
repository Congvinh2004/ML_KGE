"""
Baseline TransE Model - Tiêu chuẩn để so sánh

Code này train TransE với hyperparameters đã được tối ưu trên FB15K237
Đạt hiệu suất tốt: MRR: 0.2901, MR: 102.2, Hits@10: 0.4895

Hyperparameters:
- dim: 300
- margin: 6.0
- train_times: 1500
- alpha: 0.5
- nbatches: 50
- neg_ent: 10
- type_constrain: True

Kết quả trên FB15K237 (baseline):
- MRR: 0.2901
- MR: 102.2
- Hits@10: 0.4895
- Hits@3: 0.3195
- Hits@1: 0.1915

Đây là baseline để so sánh với các mô hình cải tiến sau này.

===========================================
CELL 1: Clone OpenKE repository
===========================================
"""

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
subprocess.run(['cp', '-r', 'OpenKE/benchmarks/FB15K237', 'benchmarks/'])

print("✅ Files copied!")
print(f"✅ OpenKE path: {os.getcwd()}/openke")
print(f"✅ Dataset path: {os.getcwd()}/benchmarks/FB15K237")

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
# CELL 6: Setup và Configuration - BASELINE
# ============================================
# Path to dataset
DATASET_PATH = "./benchmarks/FB15K237/"
CHECKPOINT_PATH = "./baseline/checkpoints"

# Create checkpoint directory
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# ⭐ BASELINE HYPERPARAMETERS - Không thay đổi để giữ làm tiêu chuẩn
print("="*60)
print("⚙️  BASELINE CONFIGURATION")
print("="*60)

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

print("Baseline Hyperparameters:")
for key, value in BASELINE_CONFIG.items():
    print(f"   {key}: {value}")

print("\n📊 Expected Results (Baseline Performance):")
print("   MRR: 0.2901")
print("   MR: 102.2")
print("   Hits@10: 0.4895")
print("   Hits@3: 0.3195")
print("   Hits@1: 0.1915")

# ============================================
# CELL 7: Load data
# ============================================
# Training data loader (sử dụng PyTorchLoader cho GPU)
train_dataloader = PyTorchTrainDataLoader(
    in_path = DATASET_PATH, 
    nbatches = BASELINE_CONFIG['nbatches'],
    threads = BASELINE_CONFIG['threads'],  # Linux support multiprocessing tốt
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = BASELINE_CONFIG['neg_ent'],
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
# CELL 8: Define Baseline TransE model
# ============================================
# Define TransE model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = BASELINE_CONFIG['dim'],  # Embedding dimension
    p_norm = BASELINE_CONFIG['p_norm'],  # L1 distance
    norm_flag = BASELINE_CONFIG['norm_flag']  # Normalize embeddings
)

print("✅ Model defined!")

# Define loss function (Negative Sampling)
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = BASELINE_CONFIG['margin']),
    batch_size = train_dataloader.get_batch_size()
)

print("✅ Loss function configured!")

# ============================================
# CELL 9: Train Baseline Model
# ============================================
print("🚀 Starting baseline training with GPU...")
print("="*50)

# Trainer với GPU
trainer = Trainer(
    model = model, 
    data_loader = train_dataloader, 
    train_times = BASELINE_CONFIG['train_times'],  # 1500 epochs
    alpha = BASELINE_CONFIG['alpha'],  # Learning rate
    use_gpu = True  # ⚡ Sử dụng GPU!
)

# Bắt đầu training
trainer.run()

# Save checkpoint
checkpoint_path = f'{CHECKPOINT_PATH}/baseline_transe.ckpt'
transe.save_checkpoint(checkpoint_path)
print("="*50)
print("✅ Training complete! Checkpoint saved.")
print(f"✅ Checkpoint: {checkpoint_path}")

# ============================================
# CELL 10: Test Baseline Model
# ============================================
print("🧪 Testing baseline model...")
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
tester.run_link_prediction(type_constrain = BASELINE_CONFIG['type_constrain'])

print("="*50)
print("✅ Testing complete!")

# ============================================
# CELL 11: Final Results
# ============================================
print("\n" + "="*60)
print("📊 BASELINE RESULTS")
print("="*60)

mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=True)
print(f"\nMRR: {mrr:.4f}  MR: {mr:.1f}  Hits@10: {hit10:.4f}  Hits@3: {hit3:.4f}  Hits@1: {hit1:.4f}")

print("\n" + "="*60)
print("✅ Baseline training completed!")
print("="*60)
print(f"\n📝 This baseline will be used to compare with improved models.")
print(f"📥 Checkpoint saved: {checkpoint_path}")
