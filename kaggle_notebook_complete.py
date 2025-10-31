# -*- coding: utf-8 -*-
"""
OpenKE TransE Training on Kaggle
Complete notebook - Copy and paste to Kaggle!

Instructions:
1. Create new Kaggle Notebook
2. Enable GPU (Settings → Accelerator → GPU)
3. Copy ENTIRE code below into one cell
4. Run all
"""

# ============================================
# CELL 1: Setup
# ============================================
import os
import shutil
import sys

# Clone OpenKE
print("📦 Cloning OpenKE...")
!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && git checkout OpenKE-PyTorch

# Copy files
print("📋 Copying files...")
shutil.copytree('OpenKE/openke', 'openke', dirs_exist_ok=True)
os.makedirs('benchmarks', exist_ok=True)
shutil.copytree('OpenKE/benchmarks/FB15K237', 'benchmarks/FB15K237', dirs_exist_ok=True)
os.makedirs('checkpoint', exist_ok=True)

print("✅ Setup complete!")

# ============================================
# CELL 2: Import và kiểm tra GPU
# ============================================
import sys
sys.path.insert(0, '.')

from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TestDataLoader
from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA version: {torch.version.cuda}")
else:
    print("⚠️ No GPU - training will be slower")

# ============================================
# CELL 3: Load data
# ============================================
print("\n📂 Loading dataset...")

DATASET_PATH = "./benchmarks/FB15K237"
CHECKPOINT_PATH = "./checkpoint"

# Training dataloader
train_dataloader = PyTorchTrainDataLoader(
    in_path=DATASET_PATH,
    nbatches=100,
    threads=4,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=5,
    neg_rel=0
)

# Test dataloader
test_dataloader = TestDataLoader(DATASET_PATH, "link")

print(f"✅ Dataset loaded!")
print(f"   Entities: {train_dataloader.get_ent_tot()}")
print(f"   Relations: {train_dataloader.get_rel_tot()}")
print(f"   Batch size: {train_dataloader.get_batch_size()}")

# ============================================
# CELL 4: Define model
# ============================================
print("\n🏗️  Building model...")

transe = TransE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=200,
    p_norm=1,
    norm_flag=True
)

model = NegativeSampling(
    model=transe,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)

print("✅ Model ready!")

# ============================================
# CELL 5: TRAIN MODEL 🚂
# ============================================
print("\n🚀 Starting training...")
print("="*60)

trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=1000,  # 1000 epochs
    alpha=1.0,
    use_gpu=True  # ⚡ Use GPU!
)

trainer.run()

# Save checkpoint
transe.save_checkpoint(f'{CHECKPOINT_PATH}/transe.ckpt')

print("="*60)
print("✅ Training complete!")
print(f"✅ Checkpoint saved: {CHECKPOINT_PATH}/transe.ckpt")

# ============================================
# CELL 6: Test model
# ============================================
print("\n🧪 Testing model...")
print("="*60)

# Load checkpoint
transe.load_checkpoint(f'{CHECKPOINT_PATH}/transe.ckpt')

# Test
tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)

print("="*60)
print("✅ All done! Download checkpoint from output section.")



