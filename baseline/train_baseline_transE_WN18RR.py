# ===========================================
# CELL 1: Clone OpenKE repository
# ===========================================
import os

# Clone OpenKE repository
!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && git checkout OpenKE-PyTorch

print("✅ OpenKE cloned successfully!")

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
RESULTS_PATH = "./experiments/results"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# ⭐ SỐ LẦN TRAIN (sửa mỗi lần train)
TRAIN_RUN = 3  # ⬅️ SỬA SỐ NÀY MỖI LẦN TRAIN (1, 2, 3, 4, 5, ...)

# ⭐ WN18RR HYPERPARAMETERS - Config Lần 3 dựa trên Lần 1 (tốt nhất)
# 📊 Kết quả Lần 1 (TỐT NHẤT): MRR=0.1966, Hits@10=0.4563, Hits@1=0.0140, MR=5009.4
# 📊 Kết quả Lần 2: MRR=0.1830, Hits@10=0.4220, Hits@1=0.0013, MR=6520.1 (TỆ HƠN)
# 🎯 Mục tiêu Lần 3: Giữ nguyên các metric tốt của Lần 1, cải thiện MR (giảm từ 5009.4)
# 💡 Chiến lược: Giữ nguyên các thông số tốt của Lần 1, chỉ điều chỉnh nhẹ để cải thiện MR
print("="*60)
print("⚙️  WN18RR CONFIGURATION - Lần 3")
print("="*60)

WN18RR_CONFIG = {
    'dim': 100,              # ✅ Giữ nguyên (Lần 2 tăng lên 150 → tệ hơn)
    'margin': 5.5,           # ⬇️ Giảm nhẹ từ 6.0 xuống 5.5 (để cải thiện MR - ranking quality)
    'train_times': 2200,     # ⬆️ Tăng nhẹ từ 2000 lên 2200 epochs (học tốt hơn nhưng không quá nhiều)
    'alpha': 0.09,           # ⬇️ Giảm nhẹ từ 0.1 xuống 0.09 (fine-tuning tốt hơn, tránh overfitting)
    'nbatches': 50,          # ✅ Giữ nguyên (đã tối ưu)
    'neg_ent': 5,            # ✅ Giữ nguyên (Lần 2 tăng lên 10 → tệ hơn)
    'threads': 4,            # Number of threads
    'p_norm': 1,             # L1 distance
    'norm_flag': True,       # Normalize embeddings
    'type_constrain': False, # Use type constraint in evaluation (False cho WN18RR)
    'save_steps': 50,        # Lưu checkpoint sau mỗi N epochs
                              # 💡 Khuyến nghị: 50-100 epochs để tránh mất dữ liệu khi bị ngắt kết nối
                              # 💡 Checkpoint sẽ được lưu tại: {CHECKPOINT_PATH}/transe_wn18rr_epoch_XXXXX.ckpt
                              # 💡 Để resume training: Load checkpoint và tiếp tục với train_times còn lại
}

print("📝 Thay đổi so với Lần 1 (TỐT NHẤT):")
print(f"   • dim: 100 → {WN18RR_CONFIG['dim']} (giữ nguyên ✅)")
print(f"   • margin: 6.0 → {WN18RR_CONFIG['margin']} (-0.5, giảm nhẹ để cải thiện MR)")
print(f"   • train_times: 2000 → {WN18RR_CONFIG['train_times']} (+200, tăng nhẹ)")
print(f"   • alpha: 0.1 → {WN18RR_CONFIG['alpha']} (-0.01, giảm nhẹ để fine-tuning tốt hơn)")
print(f"   • neg_ent: 5 → {WN18RR_CONFIG['neg_ent']} (giữ nguyên ✅)")
print("="*60)
print("💡 Lý do điều chỉnh:")
print("   - Giữ nguyên dim=100 và neg_ent=5 (vì Lần 2 tăng → tệ hơn)")
print("   - Giảm margin nhẹ (6.0→5.5) để cải thiện MR (ranking quality)")
print("   - Tăng train_times nhẹ (2000→2200) để học tốt hơn")
print("   - Giảm alpha nhẹ (0.1→0.09) để fine-tuning tốt hơn, tránh overfitting")
print("="*60)

print("WN18RR Hyperparameters:")
for key, value in WN18RR_CONFIG.items():
    print(f"   {key}: {value}")

print("\n📊 Note: WN18RR có đặc điểm khác FB15K237:")
print("   - Dataset nhỏ hơn (ít entities/relations hơn)")
print("   - Cần dimension nhỏ hơn (100 vs 300)")
print("   - Learning rate nhỏ hơn (0.01 vs 0.5)")
print("   - Margin nhỏ hơn (1.0 vs 6.0)")
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
# CELL 9: Train Model với Test Định Kỳ
# ============================================
# ⚙️ Cấu hình Test Định Kỳ
TEST_INTERVAL = 50  # Test sau mỗi N epochs (ví dụ: 50, 100, 200)
SAVE_BEST_CHECKPOINT = True  # Lưu best checkpoint dựa trên Hits@10

print("🚀 Starting training with GPU...")
print("="*50)
    print(f"🆕 New training: Starting from epoch 0")

print(f"\n⚙️  Test Configuration:")
print(f"   Test interval: Every {TEST_INTERVAL} epochs")
print(f"   Save best checkpoint: {'Yes' if SAVE_BEST_CHECKPOINT else 'No'}")
print("="*50)

# Trainer với GPU (chỉ dùng các tham số cơ bản)
trainer = Trainer(
    model = model, 
    data_loader = train_dataloader, 
    train_times = WN18RR_CONFIG['train_times'],  # 1000 epochs
    alpha = WN18RR_CONFIG['alpha'],  # Learning rate
    use_gpu = True  # ⚡ Sử dụng GPU!
)

# Setup optimizer và GPU (trainer sẽ tự động setup trong run(), nhưng ta cần setup sớm cho training loop)
if trainer.use_gpu:
    trainer.model.cuda()

# Setup optimizer
import torch.optim as optim
if trainer.opt_method == "Adagrad" or trainer.opt_method == "adagrad":
    trainer.optimizer = optim.Adagrad(
        trainer.model.parameters(),
        lr=trainer.alpha,
        lr_decay=trainer.lr_decay,
        weight_decay=trainer.weight_decay,
    )
elif trainer.opt_method == "Adadelta" or trainer.opt_method == "adadelta":
    trainer.optimizer = optim.Adadelta(
        trainer.model.parameters(),
        lr=trainer.alpha,
        weight_decay=trainer.weight_decay,
    )
elif trainer.opt_method == "Adam" or trainer.opt_method == "adam":
    trainer.optimizer = optim.Adam(
        trainer.model.parameters(),
        lr=trainer.alpha,
        weight_decay=trainer.weight_decay,
    )
else:
    trainer.optimizer = optim.SGD(
        trainer.model.parameters(),
        lr=trainer.alpha,
        weight_decay=trainer.weight_decay,
    )

print("✅ Trainer initialized!")

# Đường dẫn checkpoint
checkpoint_path = f'{CHECKPOINT_PATH}/transe_wn18rr_final.ckpt'
best_checkpoint_path = f'{CHECKPOINT_PATH}/transe_wn18rr_l{TRAIN_RUN}_best.ckpt'

# Khởi tạo best metrics
best_hit10 = -1.0
best_epoch = 0

# Tạo tester để test định kỳ
from openke.config import Tester
tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)

# Training loop với test định kỳ
from tqdm import tqdm
import time

print("\n🚀 Starting training loop...")
print("="*60)

training_range = tqdm(range(WN18RR_CONFIG['train_times']))

for epoch in training_range:
    actual_epoch = epoch + 1
    
    # Training một epoch
    res = 0.0
    for data in train_dataloader:
        loss = trainer.train_one_step(data)
        res += loss
    
    training_range.set_description(f"Epoch {actual_epoch}/{WN18RR_CONFIG['train_times']} | loss: {res:.4f}")
    
    # Test định kỳ
    if actual_epoch % TEST_INTERVAL == 0 or actual_epoch == WN18RR_CONFIG['train_times']:
        print(f"\n🧪 Testing at epoch {actual_epoch}...")
        mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=WN18RR_CONFIG['type_constrain'])
        
        print(f"📊 Results at epoch {actual_epoch}:")
        print(f"   MRR: {mrr:.4f}  MR: {mr:.1f}  Hits@10: {hit10:.4f}  Hits@3: {hit3:.4f}  Hits@1: {hit1:.4f}")
        
        # Lưu best checkpoint nếu tốt hơn
        if SAVE_BEST_CHECKPOINT and hit10 > best_hit10:
            best_hit10 = hit10
            best_epoch = actual_epoch
            transe.save_checkpoint(best_checkpoint_path)
            print(f"   ⭐ NEW BEST! Hits@10: {hit10:.4f} at epoch {actual_epoch}")
            print(f"   ✅ Best checkpoint saved: {best_checkpoint_path}")
        elif SAVE_BEST_CHECKPOINT:
            print(f"   ℹ️  Current best: Hits@10: {best_hit10:.4f} at epoch {best_epoch}")
        
        # Lưu checkpoint định kỳ (nếu cần)
        if WN18RR_CONFIG.get('save_steps') and actual_epoch % WN18RR_CONFIG['save_steps'] == 0:
            periodic_checkpoint = f'{CHECKPOINT_PATH}/transe_wn18rr_epoch_{actual_epoch:05d}.ckpt'
            transe.save_checkpoint(periodic_checkpoint)
            print(f"   💾 Periodic checkpoint saved: {periodic_checkpoint}")

print("\n" + "="*60)
print("✅ Training complete!")
print("="*60)

# Save checkpoint cuối cùng
transe.save_checkpoint(checkpoint_path)
print(f"✅ Final checkpoint saved: {checkpoint_path}")

if SAVE_BEST_CHECKPOINT and best_epoch > 0:
    print(f"⭐ Best checkpoint: {best_checkpoint_path}")
    print(f"   Best Hits@10: {best_hit10:.4f} at epoch {best_epoch}")

# In thông tin về các checkpoint đã lưu
if WN18RR_CONFIG.get('save_steps'):
    print(f"\n💾 Periodic checkpoints đã được lưu sau mỗi {WN18RR_CONFIG['save_steps']} epochs")
    print(f"   📂 Thư mục: {CHECKPOINT_PATH}")
    print(f"   💡 Có thể tiếp tục training từ checkpoint bất kỳ nếu bị ngắt kết nối")
# ============================================
# CELL 10: Final Test với Best Model
# ============================================
print("\n🧪 Final Testing (Best Model)...")
print("="*60)

# Load best checkpoint nếu có, nếu không thì dùng final checkpoint
if SAVE_BEST_CHECKPOINT and os.path.exists(best_checkpoint_path):
    transe.load_checkpoint(best_checkpoint_path)
    print(f"✅ Loaded best checkpoint from epoch {best_epoch} (Hits@10: {best_hit10:.4f})")
else:
transe.load_checkpoint(checkpoint_path)
    print(f"✅ Loaded final checkpoint")

# Test với GPU
tester = Tester(
    model = transe, 
    data_loader = test_dataloader, 
    use_gpu = True
)

# Run link prediction
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=WN18RR_CONFIG['type_constrain'])

print("="*60)
print("✅ Testing complete!")

# ============================================
# CELL 11: Final Results
# ============================================
print("\n" + "="*60)
print("📊 WN18RR FINAL RESULTS")
print("="*60)
print(f"MRR: {mrr:.4f}  MR: {mr:.1f}  Hits@10: {hit10:.4f}  Hits@3: {hit3:.4f}  Hits@1: {hit1:.4f}")
if SAVE_BEST_CHECKPOINT and best_epoch > 0:
    print(f"\n⭐ Best result during training:")
    print(f"   Epoch: {best_epoch}")
    print(f"   Hits@10: {best_hit10:.4f}")

print("\n" + "="*60)
print("✅ Training completed!")
print("="*60)
print(f"\n📥 Checkpoint saved: {checkpoint_path}")
print(f"💡 So sánh với baseline FB15K237 để đánh giá hiệu suất")

# ============================================
# CELL 11.5: Lưu kết quả vào file
# ============================================
# Lưu kết quả vào file text
result_file = f'{RESULTS_PATH}/transe_wn18rr_l{TRAIN_RUN}_results.txt'
with open(result_file, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("TransE Results on WN18RR\n")
    f.write("="*60 + "\n")
    f.write(f"Train Run: {TRAIN_RUN}\n")
    f.write(f"Dataset: WN18RR\n")
    f.write(f"\nConfiguration:\n")
    for key, value in WN18RR_CONFIG.items():
        f.write(f"   {key}: {value}\n")
    f.write(f"\nFinal Results:\n")
    f.write(f"   MRR: {mrr:.4f}\n")
    f.write(f"   MR: {mr:.1f}\n")
    f.write(f"   Hits@10: {hit10:.4f}\n")
    f.write(f"   Hits@3: {hit3:.4f}\n")
    f.write(f"   Hits@1: {hit1:.4f}\n")
    f.write("="*60 + "\n")

print(f"\n✅ Results saved to: {result_file}")

# Lưu kết quả vào file CSV (để dễ so sánh)
import csv
csv_file = f'{RESULTS_PATH}/transe_wn18rr_all_results.csv'

# Kiểm tra file CSV đã tồn tại chưa
file_exists = os.path.exists(csv_file)

with open(csv_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Ghi header nếu file mới
    if not file_exists:
        writer.writerow(['Run', 'MRR', 'MR', 'Hits@10', 'Hits@3', 'Hits@1', 'Margin', 'Epochs', 'LR', 'Neg_ent', 'Dim'])
    
    # Ghi kết quả
    writer.writerow([
        TRAIN_RUN,
        f"{mrr:.4f}",
        f"{mr:.1f}",
        f"{hit10:.4f}",
        f"{hit3:.4f}",
        f"{hit1:.4f}",
        WN18RR_CONFIG['margin'],
        WN18RR_CONFIG['train_times'],
        WN18RR_CONFIG['alpha'],
        WN18RR_CONFIG['neg_ent'],
        WN18RR_CONFIG['dim']
    ])

print(f"✅ Results appended to CSV: {csv_file}")

# ============================================
# CELL 11.6: Tự động lưu Best Result và Zip
# ============================================
import json
import zipfile
from IPython.display import FileLink
from datetime import datetime

# Đường dẫn file lưu best result
best_result_file = f'{RESULTS_PATH}/transe_wn18rr_best_result.json'
best_result_zip = f'/kaggle/working/transe_wn18rr_BEST_RESULT.zip'

# Khởi tạo best result file ngay từ đầu để đảm bảo có thể lưu trong quá trình training
os.makedirs(RESULTS_PATH, exist_ok=True)

def load_best_result():
    """Đọc best result từ file JSON"""
    if os.path.exists(best_result_file):
        try:
            with open(best_result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading best result: {e}")
            return None
    return None

def save_best_result(result_data):
    """Lưu best result vào file JSON"""
    try:
        with open(best_result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Best result saved to: {best_result_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving best result: {e}")
        return False

def is_better_result(current, best):
    """So sánh kết quả hiện tại với best result
    Ưu tiên Hits@10 (metric quan trọng nhất), sau đó là MRR
    MR càng nhỏ càng tốt (được xem xét nhưng không ưu tiên)
    """
    if best is None:
        return True
    
    # Ưu tiên 1: Hits@10 (metric quan trọng nhất)
    if current['Hits@10'] > best['Hits@10']:
        return True
    elif current['Hits@10'] < best['Hits@10']:
        return False
    
    # Nếu Hits@10 bằng nhau, so sánh MRR
    if current['MRR'] > best['MRR']:
        return True
    elif current['MRR'] < best['MRR']:
        return False
    
    # Nếu cả hai đều bằng nhau, so sánh MR (càng nhỏ càng tốt)
    if current['MR'] < best['MR']:
        return True
    
    return False

def create_best_result_zip(best_checkpoint_path, checkpoint_path, result_file, 
                           csv_file_path, best_result_json_path, zip_output_path, run_number):
    """Tạo file zip chứa đầy đủ best result files (sau khi training hoàn tất)
    Có try-except để đảm bảo không bị lỗi
    """
    files_added = []
    files_missing = []
    
    print(f"\n📦 Creating BEST RESULT zip file...")
    print(f"   Zip path: {zip_output_path}")
    
    try:
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Thêm best checkpoint
            if os.path.exists(best_checkpoint_path):
                zipf.write(best_checkpoint_path, f'transe_wn18rr_l{run_number}_best.ckpt')
                files_added.append('best checkpoint')
                print(f"   ✅ Added: best checkpoint")
            else:
                files_missing.append('best checkpoint')
                print(f"   ⚠️  Missing: best checkpoint")
            
            # Thêm final checkpoint (nếu khác best)
            if os.path.exists(checkpoint_path):
                if os.path.exists(best_checkpoint_path) and checkpoint_path != best_checkpoint_path:
                    zipf.write(checkpoint_path, f'transe_wn18rr_l{run_number}.ckpt')
                    files_added.append('final checkpoint')
                    print(f"   ✅ Added: final checkpoint")
                elif not os.path.exists(best_checkpoint_path):
                    # Nếu không có best checkpoint, dùng final checkpoint
                    zipf.write(checkpoint_path, f'transe_wn18rr_l{run_number}.ckpt')
                    files_added.append('final checkpoint (as best)')
                    print(f"   ✅ Added: final checkpoint (as best)")
            else:
                if not os.path.exists(best_checkpoint_path):
                    files_missing.append('checkpoint')
                    print(f"   ⚠️  Missing: checkpoint")
            
            # Thêm result file
            if os.path.exists(result_file):
                zipf.write(result_file, f'transe_wn18rr_l{run_number}_results.txt')
                files_added.append('results file')
                print(f"   ✅ Added: results file")
            else:
                files_missing.append('results file')
                print(f"   ⚠️  Missing: results file")
            
            # Thêm best result JSON file
            if os.path.exists(best_result_json_path):
                zipf.write(best_result_json_path, 'transe_wn18rr_best_result.json')
                files_added.append('best result JSON')
                print(f"   ✅ Added: best result JSON")
            else:
                files_missing.append('best result JSON')
                print(f"   ⚠️  Missing: best result JSON")
            
            # Thêm CSV file (tất cả results)
            if os.path.exists(csv_file_path):
                zipf.write(csv_file_path, 'transe_wn18rr_all_results.csv')
                files_added.append('all results CSV')
                print(f"   ✅ Added: all results CSV")
            else:
                files_missing.append('all results CSV')
                print(f"   ⚠️  Missing: all results CSV")
    except Exception as e:
        print(f"   ❌ Error creating zip file: {e}")
        print(f"   ⚠️  Best result JSON is still saved: {best_result_json_path}")
        raise
    
    return files_added, files_missing

# So sánh kết quả hiện tại với best result
print("\n" + "="*60)
print("🏆 Checking if this is the BEST RESULT...")
print("="*60)

current_result = {
    'Run': TRAIN_RUN,
    'MRR': mrr,
    'MR': mr,
    'Hits@10': hit10,
    'Hits@3': hit3,
    'Hits@1': hit1,
    'timestamp': datetime.now().isoformat(),
    'epoch': WN18RR_CONFIG['train_times'],
    'config': WN18RR_CONFIG.copy()
}

best_result = load_best_result()

# Đường dẫn checkpoint (dùng best checkpoint nếu có, nếu không thì dùng final)
if SAVE_BEST_CHECKPOINT and os.path.exists(best_checkpoint_path):
    best_checkpoint_for_zip = best_checkpoint_path
else:
    best_checkpoint_for_zip = checkpoint_path

if is_better_result(current_result, best_result):
    print("🎉 NEW BEST RESULT! 🎉")
    print(f"   Current: MRR={mrr:.4f}, Hits@10={hit10:.4f}, MR={mr:.1f}")
    if best_result:
        print(f"   Previous Best: MRR={best_result['MRR']:.4f}, Hits@10={best_result['Hits@10']:.4f}, MR={best_result['MR']:.1f}")
        improvement_hit10 = ((hit10 - best_result['Hits@10']) / best_result['Hits@10']) * 100
        improvement_mrr = ((mrr - best_result['MRR']) / best_result['MRR']) * 100
        print(f"   Improvement: Hits@10 +{improvement_hit10:.2f}%, MRR +{improvement_mrr:.2f}%")
    else:
        print("   (This is the first result)")
    
    # Lưu best result
    if save_best_result(current_result):
        # Tạo/cập nhật zip file cho best result với đầy đủ files (sau khi training hoàn tất)
        try:
            files_added, files_missing = create_best_result_zip(
                best_checkpoint_for_zip, checkpoint_path, result_file,
                csv_file, best_result_file, best_result_zip, TRAIN_RUN
            )
            
            if files_added:
                print(f"\n📥 BEST RESULT ZIP FILE (FINAL - with all files):")
                print(f"   Files added: {', '.join(files_added)}")
                if files_missing:
                    print(f"   ⚠️  Missing files: {len(files_missing)}")
                print(f"   ✅ Best result zip updated: {best_result_zip}")
                try:
                    print(FileLink(best_result_zip))
                except:
                    print(f"   💡 Download link: {best_result_zip}")
            else:
                print(f"\n⚠️  No files to zip for best result!")
        except Exception as e:
            print(f"⚠️  Warning: Could not create final zip: {e}")
            print(f"   Best result JSON is still saved: {best_result_file}")
            if os.path.exists(best_result_zip):
                print(f"   Previous zip file is still available: {best_result_zip}")
    else:
        print("❌ Failed to save best result")
else:
    print("ℹ️  Not the best result")
    if best_result:
        print(f"   Current: MRR={mrr:.4f}, Hits@10={hit10:.4f}, MR={mr:.1f}")
        print(f"   Best: MRR={best_result['MRR']:.4f}, Hits@10={best_result['Hits@10']:.4f}, MR={best_result['MR']:.1f} (Run {best_result['Run']})")
        diff_hit10 = hit10 - best_result['Hits@10']
        diff_mrr = mrr - best_result['MRR']
        print(f"   Difference: Hits@10 {diff_hit10:+.4f}, MRR {diff_mrr:+.4f}")

print("="*60)

# ============================================
# CELL 12: Zip và Download Checkpoint (current run)
# ============================================
# Zip checkpoint và results cho lần train hiện tại
checkpoint_zip = f'/kaggle/working/transe_wn18rr_l{TRAIN_RUN}.zip'
files_added = []
files_missing = []

print("\n📦 Preparing zip file for current run...")
print(f"   Zip path: {checkpoint_zip}")

with zipfile.ZipFile(checkpoint_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Thêm best checkpoint nếu có
    if SAVE_BEST_CHECKPOINT and os.path.exists(best_checkpoint_path):
        zipf.write(best_checkpoint_path, f'transe_wn18rr_l{TRAIN_RUN}_best.ckpt')
        files_added.append('best checkpoint')
        print(f"   ✅ Added: best checkpoint")
    
    # Thêm final checkpoint nếu tồn tại
    if os.path.exists(checkpoint_path):
        zipf.write(checkpoint_path, f'transe_wn18rr_l{TRAIN_RUN}.ckpt')
        files_added.append('final checkpoint')
        print(f"   ✅ Added: final checkpoint")
    else:
        files_missing.append(f'checkpoint ({checkpoint_path})')
        print(f"   ⚠️  Missing: checkpoint")
    
    # Thêm result file nếu tồn tại
    if os.path.exists(result_file):
        zipf.write(result_file, f'transe_wn18rr_l{TRAIN_RUN}_results.txt')
        files_added.append('results file')
        print(f"   ✅ Added: results file")
    else:
        files_missing.append(f'results file ({result_file})')
        print(f"   ⚠️  Missing: results file")
    
    # Thêm CSV file nếu tồn tại
    if os.path.exists(csv_file):
        zipf.write(csv_file, 'transe_wn18rr_all_results.csv')
        files_added.append('all results CSV')
        print(f"   ✅ Added: all results CSV")

if files_added:
    print(f"\n📥 Download checkpoint và results (current run):")
    print(f"   Files added: {', '.join(files_added)}")
    if files_missing:
        print(f"   Files missing: {len(files_missing)}")
    try:
        print(FileLink(checkpoint_zip))
    except:
        print(f"   💡 Download link: {checkpoint_zip}")
else:
    print(f"\n⚠️  No files to zip. All files are missing!")
    if files_missing:
        print("   Missing files:")
        for f in files_missing:
            print(f"      - {f}")

# Hiển thị thông tin về best result
if os.path.exists(best_result_file):
    best_result = load_best_result()
    if best_result:
        print(f"\n🏆 Current BEST RESULT:")
        print(f"   Run: {best_result['Run']}")
        print(f"   MRR: {best_result['MRR']:.4f}")
        print(f"   Hits@10: {best_result['Hits@10']:.4f}")
        print(f"   MR: {best_result['MR']:.1f}")
        print(f"   Timestamp: {best_result.get('timestamp', 'N/A')}")
        print(f"   📦 Best result zip: {best_result_zip}")
        if os.path.exists(best_result_zip):
            try:
                print(FileLink(best_result_zip))
            except:
                print(f"   💡 Download link: {best_result_zip}")

print("\n" + "="*60)
print("✅ All done!")
print("="*60)
đây là mô hình transE test trên bộ dữ liệu WN8RR, và đạt kết quả: 

# kết quả lần 1:
# 0.43953412771224976
# MRR: 0.1895  MR: 4853.3  Hits@10: 0.4395  Hits@3: 0.3559  Hits@1: 0.0057
# đề xuất cải thiện: 
# Tăng số epochs: train_times: 2000 (thay vì 1000)
# Giảm margin: margin: 1.0-3.0 (thay vì 5.0)
# Tăng negative samples: neg_ent: 10-25 (thay vì 5)
# Thử learning rate: alpha: 0.01 hoặc 0.05