"""
Kaggle Notebook script để train TransE trên FB15K237
Chạy trong Kaggle Notebook với GPU enabled
"""

# Cell 1: Setup
import sys
import os

# Nếu đã upload openke và benchmarks vào Kaggle Input datasets
sys.path.insert(0, '/kaggle/input/openke-pytorch')  # Thay bằng tên dataset của bạn

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TestDataLoader
from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader

# Cell 2: Setup data paths
# data_path = "/kaggle/input/fb15k237"  # Nếu upload dataset riêng
# Hoặc dùng copy vào working directory
os.system("cp -r /kaggle/input/fb15k237 ./benchmarks/ 2>/dev/null || echo 'Already copied'")

# Cell 3: Train model
# Path to dataset
DATASET_PATH = "/kaggle/input/fb15k237"  # Hoặc "./benchmarks/FB15K237/"
CHECKPOINT_PATH = "/kaggle/working/checkpoint"

# Tạo thư mục checkpoint
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Dataloader với GPU và multiprocessing (Linux không bị lỗi)
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

test_dataloader = TestDataLoader(DATASET_PATH, "link")

# Define model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 200, 
    p_norm = 1, 
    norm_flag = True
)

# Loss function
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = 5.0),
    batch_size = train_dataloader.get_batch_size()
)

# Train với GPU (use_gpu = True trên Kaggle)
print("Starting training with GPU...")
trainer = Trainer(
    model = model, 
    data_loader = train_dataloader, 
    train_times = 1000,  # Full training
    alpha = 1.0, 
    use_gpu = True  # Sử dụng GPU trên Kaggle
)
trainer.run()

# Save checkpoint
transe.save_checkpoint(f'{CHECKPOINT_PATH}/transe.ckpt')
print("Training complete! Checkpoint saved.")

# Test model
transe.load_checkpoint(f'{CHECKPOINT_PATH}/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("All done! Check output above for results.")



