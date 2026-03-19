# ============================================
# CELL 1: Clone OpenKE
# ============================================
!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && git checkout OpenKE-PyTorch

# ============================================
# CELL 2: Copy files
# ============================================
import shutil
import subprocess
import os

# Copy openke module
print("📂 Copying openke module...")
shutil.copytree('OpenKE/openke', 'openke', dirs_exist_ok=True)
print("✅ openke module copied!")

# Create benchmarks directory
print("📂 Creating benchmarks directory...")
os.makedirs('benchmarks', exist_ok=True)
print("✅ benchmarks directory created!")

# Copy FB15K237 dataset
print("📂 Copying FB15K237 dataset...")
subprocess.run(['cp', '-r', 'OpenKE/benchmarks/FB15K237', 'benchmarks/'], 
               check=True, 
               stdout=subprocess.DEVNULL, 
               stderr=subprocess.DEVNULL)
print("✅ FB15K237 dataset copied!")

print("\n✅ All files copied successfully!")

# ============================================
# CELL 2.5: Copy Trainer.py đã được cập nhật (Optional)
# ============================================
# 💡 Mục đích: Copy Trainer.py đã có checkpoint features vào workspace
# 
# CÓ 2 CÁCH:
# 1. Từ Kaggle Dataset (Khuyến nghị - không cần code dài)
# 2. Tạo file trực tiếp trong notebook (nếu không có dataset)
#
# ⚠️ Nếu không làm gì, sẽ dùng patch method trong CELL 5.5

import shutil
import os

# ============================================
# CÁCH 1: Copy từ Kaggle Dataset (Khuyến nghị)
# ============================================
# Bước 1: Tạo Kaggle Dataset chứa file Trainer.py
# Bước 2: Add dataset vào notebook
# Bước 3: Sửa tên dataset bên dưới

TRAINER_DATASET_NAME = "openke-trainer-updated"  # ⬅️ Sửa tên dataset nếu khác
trainer_dataset_path = f'/kaggle/input/{TRAINER_DATASET_NAME}'
trainer_source = os.path.join(trainer_dataset_path, "Trainer.py")
trainer_target = "openke/config/Trainer.py"

TRAINER_ALREADY_UPDATED = False

if os.path.exists(trainer_source):
    print("📂 Copying Trainer.py from Kaggle Dataset...")
    shutil.copy(trainer_source, trainer_target)
    print("✅ Trainer.py đã được cập nhật với checkpoint features!")
    print("   → Không cần patch trong CELL 5.5 nữa")
    TRAINER_ALREADY_UPDATED = True
else:
    print(f"⚠️  Trainer.py không tìm thấy trong dataset '{TRAINER_DATASET_NAME}'")
    print("💡 Có thể:")
    print("   1. Tạo Kaggle Dataset chứa Trainer.py và add vào notebook")
    print("   2. Hoặc dùng patch method trong CELL 5.5 (tự động)")
    print("   3. Hoặc uncomment CÁCH 2 bên dưới để tạo file trực tiếp")
    
    # ============================================
    # CÁCH 2: Tạo file trực tiếp (Uncomment nếu cần)
    # ============================================
    # ⚠️ Code này sẽ tạo file Trainer.py trực tiếp trong notebook
    # 💡 Chỉ dùng nếu không có Kaggle Dataset
    
    # Uncomment đoạn code bên dưới để bật CÁCH 2:
    # (Code sẽ được thêm vào file hướng dẫn riêng vì quá dài)
    
    # Xem file: HUONG_DAN_COPY_TRAINER_TO_WORKSPACE.md để có code đầy đủ

# ============================================
# CELL 3: Copy TTM module từ Kaggle Dataset
# ============================================
import shutil
import zipfile
import os

# Đường dẫn dataset (tên dataset của bạn)
DATASET_NAME = "ttm-module-v1"  # ⬅️ Sửa tên dataset nếu khác
dataset_path = f'/kaggle/input/{DATASET_NAME}'

print(f"📂 Dataset path: {dataset_path}")

# Kiểm tra dataset có tồn tại không
if not os.path.exists(dataset_path):
    print(f"❌ Dataset không tồn tại: {dataset_path}")
    print("💡 Hãy đảm bảo đã Add Dataset vào notebook!")
else:
    print(f"✅ Dataset found!")
    
    # Liệt kê files trong dataset
    files = os.listdir(dataset_path)
    print(f"📦 Files in dataset: {files}")
    
    # Tìm file zip hoặc thư mục ttm
    zip_file = None
    ttm_dir = None
    
    for item in files:
        item_path = os.path.join(dataset_path, item)
        if item.endswith('.zip'):
            zip_file = item_path
            print(f"📦 Found zip file: {item}")
        elif os.path.isdir(item_path) and 'ttm' in item.lower():
            ttm_dir = item_path
            print(f"📁 Found ttm directory: {item}")
    
    # Copy TTM module
    if zip_file:
        # Extract từ zip
        print(f"\n📦 Extracting {zip_file}...")
        extract_path = '/kaggle/working/ttm_temp'
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Tìm thư mục ttm trong extracted files
        ttm_source = None
        for root, dirs, files in os.walk(extract_path):
            if 'ttm' in dirs:
                ttm_source = os.path.join(root, 'ttm')
                break
        
        if not ttm_source:
            # Nếu không tìm thấy, có thể ttm ở root
            ttm_source = os.path.join(extract_path, 'ttm')
        
        if os.path.exists(ttm_source):
            # Copy vào research/ttm
            target_path = '/kaggle/working/research/ttm'
            shutil.copytree(ttm_source, target_path, dirs_exist_ok=True)
            print(f"✅ TTM module copied to: {target_path}")
        else:
            print(f"⚠️  Không tìm thấy thư mục ttm trong zip")
            print(f"   Extracted to: {extract_path}")
            print(f"   Contents: {os.listdir(extract_path)}")
    
    elif ttm_dir:
        # Copy trực tiếp từ thư mục
        target_path = '/kaggle/working/research/ttm'
        shutil.copytree(ttm_dir, target_path, dirs_exist_ok=True)
        print(f"✅ TTM module copied to: {target_path}")
    
    else:
        print("⚠️  Không tìm thấy file zip hoặc thư mục ttm trong dataset")
        print(f"   Available files: {files}")

# Kiểm tra kết quả
ttm_path = '/kaggle/working/research/ttm'
if os.path.exists(ttm_path):
    print(f"\n✅ TTM module ready at: {ttm_path}")
    print(f"   Files: {os.listdir(ttm_path)}")
else:
    print(f"\n❌ TTM module not found at: {ttm_path}")
    print("💡 Hãy kiểm tra lại dataset và đường dẫn!")

# ============================================
# CELL 4: Build Base.so
# ============================================
!apt-get -y update && apt-get -y install build-essential
%cd openke
!bash make.sh
%cd ..

# ============================================
# CELL 5: Import
# ============================================
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'research')

print("📦 Importing OpenKE modules...")
try:
    from openke.config import Trainer, Tester
    from openke.data import TestDataLoader
    from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader
    print("✅ OpenKE modules imported successfully!")
    print("   - Trainer, Tester")
    print("   - TestDataLoader")
    print("   - PyTorchTrainDataLoader")
except Exception as e:
    print(f"❌ Error importing OpenKE: {e}")
    raise

print("\n📦 Importing TTM modules...")
try:
    from research.ttm import TransE_TTM, TrustinessCalculator, TTM_Loss, TTM_NegativeSampling
    print("✅ TTM modules imported successfully!")
    print("   - TransE_TTM")
    print("   - TrustinessCalculator")
    print("   - TTM_Loss")
    print("   - TTM_NegativeSampling")
except Exception as e:
    print(f"❌ Error importing TTM: {e}")
    print("💡 Make sure TTM module is in /kaggle/working/research/ttm/")
    raise

print("\n✅ All imports successful!")

# ============================================
# CELL 5.5: Patch Trainer để hỗ trợ checkpoint định kỳ và resume
# ============================================
# ⚠️ QUAN TRỌNG: OpenKE gốc chưa có tính năng checkpoint định kỳ
# Cell này sẽ patch Trainer class để thêm các tính năng mới

import importlib
import inspect
import torch
import torch.optim as optim
import json
import datetime
from tqdm import tqdm

# Kiểm tra xem Trainer đã có các tham số mới chưa
trainer_init = inspect.signature(Trainer.__init__)
has_checkpoint_prefix = 'checkpoint_prefix' in trainer_init.parameters

# Nếu Trainer đã được cập nhật từ dataset (CELL 2.5), không cần patch
if 'TRAINER_ALREADY_UPDATED' in globals() and TRAINER_ALREADY_UPDATED:
    print("✅ Trainer.py đã được cập nhật từ dataset (no patch needed)")
    has_checkpoint_prefix = True  # Giả lập để bỏ qua patch

if not has_checkpoint_prefix:
    print("🔧 Patching Trainer class to add checkpoint features...")
    
    # Lưu lại các method gốc
    original_run = Trainer.run
    original_init = Trainer.__init__
    
    def patched_init(self, 
                     model = None,
                     data_loader = None,
                     train_times = 1000,
                     alpha = 0.5,
                     use_gpu = True,
                     opt_method = "sgd",
                     save_steps = None,
                     checkpoint_dir = None,
                     checkpoint_prefix = None,
                     save_model = None,
                     resume_from_checkpoint = None,
                     resume_from_latest = False):
        # Gọi __init__ gốc với các tham số cơ bản (chỉ những tham số mà Trainer gốc có)
        # Trainer gốc có: model, data_loader, train_times, alpha, use_gpu, opt_method
        original_init(self, model, data_loader, train_times, alpha, use_gpu, opt_method)
        # Thêm các thuộc tính mới (có thể Trainer gốc không có)
        if not hasattr(self, 'save_steps'):
            self.save_steps = save_steps
        if not hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.save_model = save_model
        self.resume_from_checkpoint = resume_from_checkpoint
        self.resume_from_latest = resume_from_latest
        self.start_epoch = 0
        self.original_train_times = train_times
    
    def patched_save_checkpoint(self, epoch, loss):
        """Lưu checkpoint với tên file và metadata"""
        if not self.checkpoint_dir:
            return
        
        import datetime
        import json
        
        # Tạo tên file checkpoint
        if self.checkpoint_prefix:
            checkpoint_name = f"{self.checkpoint_prefix}_epoch_{epoch:05d}.ckpt"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch:05d}.ckpt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Lưu model
        model_to_save = self.save_model if self.save_model is not None else self.model
        if hasattr(model_to_save, 'save_checkpoint'):
            model_to_save.save_checkpoint(checkpoint_path)
        else:
            torch.save(model_to_save.state_dict(), checkpoint_path)
        
        # Lưu metadata
        metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
        metadata = {
            'epoch': epoch,
            'loss': float(loss),
            'timestamp': datetime.datetime.now().isoformat(),
            'train_times': self.original_train_times,
            'alpha': self.alpha,
            'opt_method': self.opt_method
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"   ✅ Checkpoint saved: {checkpoint_name} ({file_size:.2f} MB)")
        print(f"   📄 Metadata: {os.path.basename(metadata_path)}")
    
    def patched_find_latest_checkpoint(self):
        """Tìm checkpoint mới nhất"""
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return None, None
        
        checkpoint_files = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(self.checkpoint_prefix) and filename.endswith('.ckpt'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                metadata_path = filepath.replace('.ckpt', '_metadata.json')
                epoch = 0
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            epoch = meta.get('epoch', 0)
                    except:
                        pass
                else:
                    import re
                    match = re.search(r'epoch_(\d+)', filename)
                    if match:
                        epoch = int(match.group(1))
                checkpoint_files.append((filepath, epoch))
        
        if not checkpoint_files:
            return None, None
        
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        latest_checkpoint_path, latest_epoch = checkpoint_files[0]
        
        metadata_path = latest_checkpoint_path.replace('.ckpt', '_metadata.json')
        metadata = {'epoch': latest_epoch, 'loss': 0.0}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        return latest_checkpoint_path, metadata
    
    def patched_load_checkpoint_for_resume(self):
        """Load checkpoint để resume"""
        checkpoint_path = None
        metadata = None
        
        if self.resume_from_checkpoint:
            checkpoint_path = self.resume_from_checkpoint
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")
                return None, None
        elif self.resume_from_latest and self.checkpoint_dir and self.checkpoint_prefix:
            checkpoint_path, metadata = self.patched_find_latest_checkpoint()
            if not checkpoint_path:
                print(f"⚠️  No checkpoint found in {self.checkpoint_dir}")
                return None, None
        
        if not checkpoint_path:
            return None, None
        
        metadata_path = checkpoint_path.replace('.ckpt', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            import re
            match = re.search(r'epoch_(\d+)', checkpoint_path)
            if match:
                metadata = {'epoch': int(match.group(1)), 'loss': 0.0}
            else:
                print(f"⚠️  Cannot determine epoch from checkpoint: {checkpoint_path}")
                return None, None
        
        model_to_load = self.save_model if self.save_model is not None else self.model
        if hasattr(model_to_load, 'load_checkpoint'):
            model_to_load.load_checkpoint(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cuda' if self.use_gpu else 'cpu')
            model_to_load.load_state_dict(state_dict)
        
        return checkpoint_path, metadata
    
    def patched_run(self):
        """Run với hỗ trợ resume và checkpoint định kỳ"""
        if self.use_gpu:
            self.model.cuda()
        
        # Resume từ checkpoint nếu có
        if self.resume_from_latest or self.resume_from_checkpoint:
            checkpoint_path, metadata = self.patched_load_checkpoint_for_resume()
            if checkpoint_path:
                print(f"✅ Resumed from checkpoint: {os.path.basename(checkpoint_path)}")
                print(f"   📊 Epoch: {metadata['epoch']}, Loss: {metadata['loss']:.4f}")
                self.start_epoch = metadata['epoch']
                target_epochs = getattr(self, 'original_train_times', self.train_times)
                remaining_epochs = target_epochs - self.start_epoch
                if remaining_epochs > 0:
                    print(f"   ⏳ Remaining epochs: {remaining_epochs} (from {self.start_epoch} to {target_epochs})")
                    self.train_times = remaining_epochs
                else:
                    print(f"   ⚠️  Training already completed! (Epoch {self.start_epoch}/{target_epochs})")
                    return
        
        # Khởi tạo optimizer (giữ nguyên code gốc)
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr = self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")
        
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            actual_epoch = self.start_epoch + epoch + 1
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (actual_epoch, res))
            
            # Lưu checkpoint định kỳ
            if self.save_steps and self.checkpoint_dir and actual_epoch % self.save_steps == 0:
                print("\n💾 Epoch %d has finished, saving checkpoint..." % (actual_epoch))
                self.patched_save_checkpoint(actual_epoch, res)
    
    # Patch các method
    Trainer.__init__ = patched_init
    Trainer.run = patched_run
    Trainer._save_checkpoint = patched_save_checkpoint
    Trainer._find_latest_checkpoint = patched_find_latest_checkpoint
    Trainer._load_checkpoint_for_resume = patched_load_checkpoint_for_resume
    
    print("✅ Trainer class patched successfully!")
    print("   - Added checkpoint_prefix, save_model parameters")
    print("   - Added resume_from_checkpoint, resume_from_latest parameters")
    print("   - Added automatic checkpoint saving every N epochs")
    print("   - Added resume training functionality")
else:
    print("✅ Trainer already has checkpoint features (no patch needed)")

# ============================================
# CELL 6: Config
# ============================================
print("⚙️  Setting up configuration...")

DATASET_PATH = "./benchmarks/FB15K237/"
CHECKPOINT_PATH = "./research/ttm/checkpoints"
RESULTS_PATH = "./research/ttm/results"

# Tạo các thư mục cần thiết
print(f"📂 Creating directories...")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
print(f"   ✅ Checkpoint path: {CHECKPOINT_PATH}")
print(f"   ✅ Results path: {RESULTS_PATH}")

# Kiểm tra dataset có tồn tại không
if not os.path.exists(DATASET_PATH):
    print(f"⚠️  Warning: Dataset path not found: {DATASET_PATH}")
    print("   Make sure CELL 2 ran successfully!")
else:
    print(f"   ✅ Dataset path: {DATASET_PATH}")

# ⭐ SỐ LẦN TRAIN (sửa mỗi lần train)
TRAIN_RUN = 1  # ⬅️ SỬA SỐ NÀY MỖI LẦN TRAIN (1, 2, 3, ...)

# TransT Configuration cho FB15K237 - VERSION 2 (Tối ưu hơn)
# 
# 📊 So sánh với config cũ:
# - dim: 200 → 300 (tăng để match baseline)
# - margin: 5.0 → 6.0 (tăng để match baseline)
# - train_times: 1000 → 1500 (tăng epochs để match baseline)
# - alpha: 1.0 → 0.5 (giảm learning rate để match baseline)
# - nbatches: 100 → 50 (giảm để match baseline, nhưng giữ nguyên nếu muốn train nhanh hơn)
# - type_constrain: False → True (bật type constraint như baseline)
# - use_cross_entropy: True → False (thử margin loss như baseline)
# - alpha_trust: 0.5 → 0.3 (giảm weight type-based)
# - beta_trust: 0.5 → 0.7 (tăng weight description-based)
#
# ⚠️ LƯU Ý: Config này sẽ train lâu hơn (~13-14 giờ trên Kaggle)
# Nếu thời gian vượt quá, có thể giảm train_times xuống 1200-1300

# ============================================
# CONFIG VERSION 2: Match Baseline (Recommended)
# ============================================
TRANST_CONFIG = {
    'dim': 300,              # Embedding dimension (tăng từ 200 → 300 để match baseline)
    'margin': 6.0,           # Margin for loss (tăng từ 5.0 → 6.0 để match baseline)
    'train_times': 1500,     # Number of epochs (tăng từ 1000 → 1500 để match baseline)
    'alpha': 0.5,            # Learning rate (giảm từ 1.0 → 0.5 để match baseline)
    'nbatches': 50,          # Number of batches per epoch (giảm từ 100 → 50 để match baseline)
    'neg_ent': 10,           # Negative samples per positive (giữ nguyên)
    'threads': 0,            # Number of threads (0 = single thread, tốt cho Windows/Kaggle)
    'p_norm': 1,             # L1 distance
    'norm_flag': True,       # Normalize embeddings
    'type_constrain': True,  # Use type constraint in evaluation (bật như baseline)
    
    # TTM specific - Điều chỉnh trustiness weights
    'use_trustiness': True,  # Có dùng trustiness không
    'alpha_trust': 0.3,      # Weight cho type-based trustiness (giảm từ 0.5 → 0.3)
    'beta_trust': 0.7,       # Weight cho description-based trustiness (tăng từ 0.5 → 0.7)
    'use_cross_entropy': False,  # Dùng margin loss (False) như baseline thay vì cross-entropy
    
    # Checkpoint settings - Lưu checkpoint định kỳ để tránh mất dữ liệu khi bị ngắt kết nối
    'save_steps': 100,       # Lưu checkpoint sau mỗi N epochs (None để tắt, khuyến nghị: 50-100)
                              # 💡 Checkpoint sẽ được lưu tại: {CHECKPOINT_PATH}/transt_fb15k237_l{TRAIN_RUN}_epoch_XXXXX.ckpt
                              # 💡 Mỗi checkpoint đi kèm file metadata chứa epoch, loss, timestamp
                              # 💡 Để resume training: Load checkpoint và tiếp tục với train_times còn lại
}

# ============================================
# CONFIG VERSION 3: Fast Training (Tối ưu thời gian - 6-7 giờ)
# ============================================
# ⚡ Config này tối ưu thời gian training (giảm từ 10+ giờ xuống 6-7 giờ):
# - Giữ dim=300, margin=6.0 như baseline
# - Giảm train_times xuống 1200 (giảm 20%)
# - Tăng nbatches lên 100 (tăng 100% - quan trọng nhất!)
# - Giảm neg_ent xuống 8 (giảm 20%)
# - Thời gian: ~6-7 giờ (giảm 30-40%)
# - Chất lượng: Vẫn tốt (có thể giảm nhẹ 1-2%)
#
# TRANST_CONFIG = {
#     'dim': 300,              # Embedding dimension (match baseline)
#     'margin': 6.0,           # Margin for loss (match baseline)
#     'train_times': 1200,     # Number of epochs (giảm 20% để train nhanh hơn)
#     'alpha': 0.5,            # Learning rate (match baseline)
#     'nbatches': 100,         # Number of batches per epoch (tăng 100% - tối ưu nhất!)
#     'neg_ent': 8,            # Negative samples (giảm 20% để train nhanh hơn)
#     'threads': 0,            # Number of threads
#     'p_norm': 1,             # L1 distance
#     'norm_flag': True,       # Normalize embeddings
#     'type_constrain': True,  # Use type constraint in evaluation
#     
#     # TTM specific
#     'use_trustiness': True,  # Có dùng trustiness không
#     'alpha_trust': 0.3,      # Weight cho type-based trustiness
#     'beta_trust': 0.7,       # Weight cho description-based trustiness
#     'use_cross_entropy': False,  # Dùng margin loss như baseline
#     
#     # Checkpoint settings
#     'save_steps': 100,
# }

# ============================================
# CONFIG VERSION 4: Very Fast (5-6 giờ) - Cần test
# ============================================
# ⚡ Config này tối ưu tối đa thời gian (giảm từ 10+ giờ xuống 5-6 giờ):
# - Giảm train_times xuống 1000 (giảm 33%)
# - Tăng nbatches lên 100
# - Giảm neg_ent xuống 5 (giảm 50%)
# - Tăng alpha lên 0.6 (tăng 20%)
# - Thời gian: ~5-6 giờ (giảm 40-50%)
# - ⚠️ Cần test để đảm bảo chất lượng
#
# TRANST_CONFIG = {
#     'dim': 300,
#     'margin': 6.0,
#     'train_times': 1000,     # Giảm 33%
#     'alpha': 0.6,            # Tăng nhẹ 20%
#     'nbatches': 100,         # Tăng 100%
#     'neg_ent': 5,            # Giảm 50%
#     'threads': 0,
#     'p_norm': 1,
#     'norm_flag': True,
#     'type_constrain': True,
#     'use_trustiness': True,
#     'alpha_trust': 0.3,
#     'beta_trust': 0.7,
#     'use_cross_entropy': False,
#     'save_steps': 100,
# }

print("\n" + "="*60)
print("⚙️  TransT (TransE + Triple Trustiness) Configuration")
print("="*60)
print(f"📊 Train Run: {TRAIN_RUN}")
print(f"📂 Dataset: FB15K237")
print("\n📋 Hyperparameters:")
for key, value in TRANST_CONFIG.items():
    print(f"   {key:20s}: {value}")

# Tính toán thời gian ước tính
total_iterations = TRANST_CONFIG['train_times'] * TRANST_CONFIG['nbatches']
print("\n⏱️  Thông tin training:")
print(f"   Tổng iterations: {total_iterations:,} ({TRANST_CONFIG['train_times']} epochs × {TRANST_CONFIG['nbatches']} batches)")
print(f"   ⚠️  Lưu ý: Thời gian thực tế có thể khác với ước tính ban đầu")
print(f"   💡 Nếu vượt quá 9 giờ, có thể giảm train_times hoặc nbatches")
print("="*60)

# ============================================
# CELL 7: Load data
# ============================================
train_dataloader = PyTorchTrainDataLoader(
    in_path=DATASET_PATH,
    nbatches=TRANST_CONFIG['nbatches'],
    threads=TRANST_CONFIG['threads'],
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=TRANST_CONFIG['neg_ent'],
    neg_rel=0
)

test_dataloader = TestDataLoader(DATASET_PATH, "link")

print(f"✅ Data loaded: {train_dataloader.get_ent_tot()} entities, {train_dataloader.get_rel_tot()} relations")

# ============================================
# CELL 8: Setup Trustiness (nếu có entity types/descriptions)
# ============================================
print("📊 Setting up Trustiness Calculator...")

trustiness_calculator = TrustinessCalculator(
    dataset_path=DATASET_PATH,
    alpha=TRANST_CONFIG['alpha_trust'],
    beta=TRANST_CONFIG['beta_trust']
)

print("\n📂 Loading entity types...")
entity_types = trustiness_calculator.load_entity_types()
if entity_types:
    print(f"   ✅ Loaded {len(entity_types)} entities với types")
else:
    print(f"   ⚠️  Không tìm thấy entity types")
    print(f"      → Sẽ dùng type-based trustiness = 1.0 (default)")

print("\n📂 Loading entity descriptions...")
entity_descriptions = trustiness_calculator.load_entity_descriptions()
if entity_descriptions:
    print(f"   ✅ Loaded {len(entity_descriptions)} entities với descriptions")
else:
    print(f"   ⚠️  Không tìm thấy entity descriptions")
    print(f"      → Sẽ dùng description-based trustiness = 1.0 (default)")

# Tính trustiness cho training triples (nếu cần)
if TRANST_CONFIG['use_trustiness']:
    print("\n📊 Calculating trustiness scores for training triples...")
    print("   ⚠️  Note: Trustiness sẽ được tính on-the-fly trong quá trình training")
    print("   → Nếu muốn pre-calculate, cần load training triples trước")
    
    # Trustiness scores sẽ được tính khi cần (on-the-fly)
    trustiness_weights = {}  # Sẽ được tính khi training
    print(f"   ✅ Trustiness calculator ready!")
    print(f"      → Type-based weight (α): {TRANST_CONFIG['alpha_trust']}")
    print(f"      → Description-based weight (β): {TRANST_CONFIG['beta_trust']}")
else:
    trustiness_weights = {}
    print("\n⚠️  Trustiness disabled (use_trustiness=False)")
    print("   → Model sẽ hoạt động như TransE thông thường")

# ============================================
# CELL 9: Create model
# ============================================
transt = TransE_TTM(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=TRANST_CONFIG['dim'],
    p_norm=TRANST_CONFIG['p_norm'],
    norm_flag=TRANST_CONFIG['norm_flag'],
    margin=TRANST_CONFIG['margin'],
    trustiness_calculator=trustiness_calculator
)

ttm_loss = TTM_Loss(
    margin=TRANST_CONFIG['margin'],
    use_cross_entropy=TRANST_CONFIG['use_cross_entropy'],
    trustiness_weights=trustiness_weights
)

model = TTM_NegativeSampling(
    model=transt,
    loss=ttm_loss,
    batch_size=train_dataloader.get_batch_size()
)

print("✅ Model created!")
print(f"   → Entity embeddings: [{train_dataloader.get_ent_tot()}, {TRANST_CONFIG['dim']}]")
print(f"   → Relation embeddings: [{train_dataloader.get_rel_tot()}, {TRANST_CONFIG['dim']}]")
print(f"   → Batch size: {train_dataloader.get_batch_size()}")

# ============================================
# CELL 10: Train và Test
# ============================================
# Cell này sẽ train model và tự động test sau khi train xong
# ⚠️ QUAN TRỌNG: Phải chạy CELL 9 trước để tạo model!
#
# 💡 RESUME TRAINING: Nếu bị ngắt kết nối (ví dụ ở epoch 800), có 2 cách để tiếp tục:
#   1. Tự động tìm checkpoint mới nhất: Set RESUME_FROM_LATEST = True
#   2. Chỉ định checkpoint cụ thể: Set RESUME_FROM_CHECKPOINT = "path/to/checkpoint.ckpt"
#
# 📝 Ví dụ: Nếu train tới epoch 800 bị ngắt, khi mở lại:
#   - Set RESUME_FROM_LATEST = True (hoặc chỉ định checkpoint epoch 800)
#   - Chạy lại CELL 9 (tạo model) và CELL 10 (train)
#   - Training sẽ tiếp tục từ epoch 801 đến 1500

# ⚙️ Cấu hình Resume (chỉnh sửa nếu cần resume)
RESUME_FROM_LATEST = False  # True để tự động tìm checkpoint mới nhất
RESUME_FROM_CHECKPOINT = None  # Đường dẫn checkpoint cụ thể (ví dụ: "./checkpoints/transt_fb15k237_l1_epoch_00800.ckpt")

print("\n🚀 Starting training...")
print("="*60)

# Kiểm tra các biến cần thiết
required_vars = ['model', 'train_dataloader', 'transt', 'test_dataloader']
missing_vars = [var for var in required_vars if var not in globals()]

if missing_vars:
    print(f"❌ Missing variables: {missing_vars}")
    print("💡 Hãy chạy các cell trước đó:")
    print("   - CELL 7: Load data (train_dataloader, test_dataloader)")
    print("   - CELL 8: Setup Trustiness")
    print("   - CELL 9: Create model (model, transt)")
    raise NameError(f"Variables not defined: {missing_vars}")

print("✅ All required variables found!")
print(f"   Dataset: FB15K237")
print(f"   Epochs: {TRANST_CONFIG['train_times']}")
print(f"   Learning rate: {TRANST_CONFIG['alpha']}")
print(f"   Batches per epoch: {TRANST_CONFIG['nbatches']}")
print(f"   Negative samples: {TRANST_CONFIG['neg_ent']}")

# Hiển thị thông tin resume
if RESUME_FROM_LATEST or RESUME_FROM_CHECKPOINT:
    print(f"\n🔄 Resume mode: ON")
    if RESUME_FROM_CHECKPOINT:
        print(f"   📂 Checkpoint: {RESUME_FROM_CHECKPOINT}")
    else:
        print(f"   🔍 Auto-find latest checkpoint in: {CHECKPOINT_PATH}")
else:
    print(f"\n🆕 New training: Starting from epoch 0")
print("="*60)

trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=TRANST_CONFIG['train_times'],
    alpha=TRANST_CONFIG['alpha'],
    use_gpu=True,
    save_steps=TRANST_CONFIG.get('save_steps'),  # Lưu checkpoint sau mỗi N epochs
    checkpoint_dir=CHECKPOINT_PATH,  # Thư mục lưu checkpoint
    checkpoint_prefix=f'transt_fb15k237_l{TRAIN_RUN}',  # Prefix cho tên file checkpoint
    save_model=transt,  # Lưu model TransE_TTM thực tế (không phải wrapper)
    resume_from_latest=RESUME_FROM_LATEST,  # Tự động tìm checkpoint mới nhất
    resume_from_checkpoint=RESUME_FROM_CHECKPOINT  # Hoặc chỉ định checkpoint cụ thể
)

trainer.run()

print("="*60)
print("✅ Training complete!")

# In thông tin về các checkpoint đã lưu
if TRANST_CONFIG.get('save_steps'):
    print(f"\n💾 Checkpoints đã được lưu sau mỗi {TRANST_CONFIG['save_steps']} epochs")
    print(f"   📂 Thư mục: {CHECKPOINT_PATH}")
    print(f"   💡 Có thể tiếp tục training từ checkpoint bất kỳ nếu bị ngắt kết nối")

# ============================================
# Save checkpoint (QUAN TRỌNG: Phải lưu trước khi test)
# ============================================
print("\n💾 Saving checkpoint...")

# Save checkpoint với số lần train
checkpoint_path = f'{CHECKPOINT_PATH}/transt_fb15k237_l{TRAIN_RUN}.ckpt'
transt.save_checkpoint(checkpoint_path)
print(f"✅ Checkpoint (với số): {checkpoint_path}")

# Cũng lưu vào file mặc định để dễ load
checkpoint_path_default = f'{CHECKPOINT_PATH}/transt_fb15k237.ckpt'
transt.save_checkpoint(checkpoint_path_default)
print(f"✅ Checkpoint (mặc định): {checkpoint_path_default}")

# Kiểm tra checkpoint đã được lưu
if os.path.exists(checkpoint_path):
    size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"   Size: {size:.2f} MB")
else:
    print(f"⚠️  Warning: Checkpoint file not found after saving!")

# ============================================
# Test Model (tự động sau khi train xong)
# ============================================
print("\n🧪 Testing model...")
print("="*60)

# Load lại checkpoint để đảm bảo test với model đã được lưu
print(f"📥 Loading checkpoint for testing: {os.path.basename(checkpoint_path)}")
try:
    transt.load_checkpoint(checkpoint_path)
    print("✅ Checkpoint loaded for testing!")
except Exception as e:
    print(f"⚠️  Warning: Could not reload checkpoint: {e}")
    print("   → Will test with current model state")

# Test
tester = Tester(model=transt, data_loader=test_dataloader, use_gpu=True)

mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=TRANST_CONFIG['type_constrain'])

print("="*60)
print("📊 TransT RESULTS on FB15K237:")
print("="*60)
print(f"MRR: {mrr:.4f}  MR: {mr:.1f}  Hits@10: {hit10:.4f}  Hits@3: {hit3:.4f}  Hits@1: {hit1:.4f}")
print("="*60)

# ============================================
# CELL 12: Lưu kết quả vào file
# ============================================
# 📝 LƯU Ý VỀ FILE KẾT QUẢ:
# - Kết quả sẽ được lưu vào: {RESULTS_PATH}/transt_fb15k237_l{TRAIN_RUN}_results.txt
# - Kết quả cũng được append vào CSV: {RESULTS_PATH}/transt_fb15k237_all_results.csv
# - Nếu resume training, kết quả cuối cùng vẫn được lưu vào cùng file (ghi đè hoặc append)
# - Checkpoint định kỳ được lưu tại: {CHECKPOINT_PATH}/transt_fb15k237_l{TRAIN_RUN}_epoch_XXXXX.ckpt
# - Metadata của mỗi checkpoint: {CHECKPOINT_PATH}/transt_fb15k237_l{TRAIN_RUN}_epoch_XXXXX_metadata.json

RESULTS_PATH = "./research/ttm/results"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Lưu kết quả vào file text
result_file = f'{RESULTS_PATH}/transt_fb15k237_l{TRAIN_RUN}_results.txt'
with open(result_file, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("TransT (TransE + Triple Trustiness) Results\n")
    f.write("="*60 + "\n")
    f.write(f"Train Run: {TRAIN_RUN}\n")
    f.write(f"Dataset: FB15K237\n")
    f.write(f"\nConfiguration:\n")
    for key, value in TRANST_CONFIG.items():
        f.write(f"   {key}: {value}\n")
    f.write(f"\nResults:\n")
    f.write(f"   MRR: {mrr:.4f}\n")
    f.write(f"   MR: {mr:.1f}\n")
    f.write(f"   Hits@10: {hit10:.4f}\n")
    f.write(f"   Hits@3: {hit3:.4f}\n")
    f.write(f"   Hits@1: {hit1:.4f}\n")
    f.write("="*60 + "\n")

print(f"\n✅ Results saved to: {result_file}")

# Lưu kết quả vào file CSV (để dễ so sánh)
import csv
csv_file = f'{RESULTS_PATH}/transt_fb15k237_all_results.csv'

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
        TRANST_CONFIG['margin'],
        TRANST_CONFIG['train_times'],
        TRANST_CONFIG['alpha'],
        TRANST_CONFIG['neg_ent'],
        TRANST_CONFIG['dim']
    ])

print(f"✅ Results appended to CSV: {csv_file}")

# ============================================
# CELL 13: Download checkpoint và results (optional)
# ============================================
import zipfile
from IPython.display import FileLink

# Zip checkpoint
checkpoint_zip = f'/kaggle/working/transt_fb15k237_l{TRAIN_RUN}.zip'
with zipfile.ZipFile(checkpoint_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(checkpoint_path, f'transt_fb15k237_l{TRAIN_RUN}.ckpt')
    zipf.write(result_file, f'transt_fb15k237_l{TRAIN_RUN}_results.txt')

print(f"\n📥 Download checkpoint và results:")
print(FileLink(checkpoint_zip))

print("\n" + "="*60)
print("✅ All done!")
print("="*60)
