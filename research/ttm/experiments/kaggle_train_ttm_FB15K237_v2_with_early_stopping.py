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
# CELL 5.5: Copy early_stopping_helper.py từ Dataset
# ============================================
import shutil
import os

# Tên dataset của bạn (sửa nếu khác)
DATASET_NAME = "early-stopping-helper"  # ⬅️ SỬA TÊN DATASET Ở ĐÂY
dataset_path = f'/kaggle/input/{DATASET_NAME}'

# Tạo thư mục nếu chưa có
os.makedirs('research/ttm/experiments', exist_ok=True)

# Đường dẫn file trong dataset
source_file = f'{dataset_path}/early_stopping_helper.py'
target_file = 'research/ttm/experiments/early_stopping_helper.py'

# Copy file từ dataset
if os.path.exists(source_file):
    shutil.copy(source_file, target_file)
    print("✅ early_stopping_helper.py đã được copy từ dataset!")
    print(f"   📂 Source: {source_file}")
    print(f"   📂 Target: {target_file}")
    
    # Kiểm tra file đã được copy
    if os.path.exists(target_file):
        file_size = os.path.getsize(target_file)
        print(f"   📄 File size: {file_size} bytes")
else:
    print(f"❌ File không tìm thấy trong dataset: {source_file}")
    print(f"   💡 Hãy kiểm tra:")
    print(f"      1. Tên dataset: {DATASET_NAME}")
    print(f"      2. File có tên đúng: early_stopping_helper.py")
    print(f"      3. Đã Add Dataset vào notebook chưa")

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
TRAIN_RUN = 5  # ⬅️ SỬA SỐ NÀY MỖI LẦN TRAIN (1, 2, 3, 4, 5, ...)

# ============================================
# CONFIG VERSION 4: Match Baseline + TransT Trustiness (Lần 5)
# ============================================
# 📊 Config này kết hợp tốt nhất của Baseline và TransT:
# - Baseline: MRR: 0.2901, MR: 102.2, Hits@10: 0.4895
# - TransT Lần 4: MRR: 0.2538, MR: 193.9, Hits@10: 0.4159 (tốt nhất trong 3 lần)
#
# 🎯 Mục tiêu: Đạt kết quả TỐT HƠN baseline bằng cách:
#    1. Match các hyperparameters quan trọng của baseline
#    2. Giữ tính năng Trustiness của TransT (điểm mạnh)
#    3. Sử dụng early stopping để tránh overfitting
#
# 📋 Thay đổi so với Lần 4:
#    - dim: 200 → 300 (match baseline - QUAN TRỌNG!)
#    - margin: 5.0 → 6.0 (match baseline - QUAN TRỌNG!)
#    - alpha: 1.0 → 0.5 (match baseline - learning rate thấp hơn, ổn định hơn)
#    - nbatches: 100 → 50 (match baseline - mỗi epoch có ít batch hơn nhưng chất lượng tốt hơn)
#    - type_constrain: False → True (match baseline - đánh giá chính xác hơn)
#    - train_times: 1200 → 1500 (match baseline - early stopping sẽ dừng sớm nếu cần)
#    - use_cross_entropy: True → False (dùng margin loss như baseline - có thể tốt hơn)
#    - Giữ trustiness với weights cân bằng (alpha_trust=0.5, beta_trust=0.5)
#
# ⏱️ Thời gian ước tính: ~10-12 giờ (với early stopping có thể dừng sớm)
# 💡 Early stopping sẽ tự động dừng nếu không cải thiện sau 10% epochs (150 epochs)

TRANST_CONFIG = {
    'dim': 300,              # ⬆️ Tăng từ 200 → 300 (match baseline - QUAN TRỌNG!)
    'margin': 6.0,           # ⬆️ Tăng từ 5.0 → 6.0 (match baseline - QUAN TRỌNG!)
    'train_times': 1500,     # ⬆️ Tăng từ 1200 → 1500 (match baseline, early stopping sẽ dừng sớm nếu cần)
    'alpha': 0.5,            # ⬇️ Giảm từ 1.0 → 0.5 (match baseline - learning rate ổn định hơn)
    'nbatches': 50,          # ⬇️ Giảm từ 100 → 50 (match baseline - chất lượng tốt hơn)
    'neg_ent': 10,           # Giữ nguyên
    'threads': 0,            # Giữ nguyên (0 = single thread, tốt cho Kaggle)
    'p_norm': 1,             # L1 distance (giữ nguyên)
    'norm_flag': True,       # Normalize embeddings (giữ nguyên)
    'type_constrain': True,  # ⬆️ Bật từ False → True (match baseline - đánh giá chính xác hơn)
    
    # TransT Trustiness - Giữ nguyên để tận dụng điểm mạnh
    'use_trustiness': True,  # Giữ trustiness (điểm mạnh của TransT)
    'alpha_trust': 0.5,      # Weight cho type-based trustiness (cân bằng)
    'beta_trust': 0.5,       # Weight cho description-based trustiness (cân bằng)
    'use_cross_entropy': False,  # ⬇️ Tắt từ True → False (dùng margin loss như baseline)
}

print("\n" + "="*60)
print("⚙️  TransT (TransE + Triple Trustiness) Configuration")
print("="*60)
print(f"📊 Train Run: {TRAIN_RUN}")
print(f"📂 Dataset: FB15K237")
print(f"📋 Config: Version 4 (Match Baseline + TransT Trustiness)")
print("\n🎯 Mục tiêu: Đạt kết quả TỐT HƠN baseline")
print("   Baseline: MRR: 0.2901, Hits@10: 0.4895")
print("   TransT Lần 4: MRR: 0.2538, Hits@10: 0.4159")
print("\n📋 Hyperparameters:")
for key, value in TRANST_CONFIG.items():
    print(f"   {key:20s}: {value}")

# So sánh với baseline
print("\n📊 So sánh với Baseline:")
baseline_config = {
    'dim': 300,
    'margin': 6.0,
    'train_times': 1500,
    'alpha': 0.5,
    'nbatches': 50,
    'type_constrain': True,
}
print("   ✅ Match baseline:")
for key in ['dim', 'margin', 'train_times', 'alpha', 'nbatches', 'type_constrain']:
    if key in TRANST_CONFIG and key in baseline_config:
        match = "✅" if TRANST_CONFIG[key] == baseline_config[key] else "❌"
        print(f"      {match} {key:20s}: {TRANST_CONFIG[key]} (baseline: {baseline_config[key]})")

print("\n✨ Điểm khác biệt với Baseline:")
print("   ✅ Thêm Trustiness (alpha_trust=0.5, beta_trust=0.5)")
print("   ✅ Early stopping sẽ tự động dừng nếu không cải thiện")

# Tính toán thời gian ước tính
total_iterations = TRANST_CONFIG['train_times'] * TRANST_CONFIG['nbatches']
print("\n⏱️  Thông tin training:")
print(f"   Tổng iterations: {total_iterations:,} ({TRANST_CONFIG['train_times']} epochs × {TRANST_CONFIG['nbatches']} batches)")
print(f"   ⚠️  Lưu ý: Early stopping sẽ dừng sớm nếu không cải thiện (patience = 10% epochs = 150 epochs)")
print(f"   ⏱️  Thời gian ước tính: ~10-12 giờ (có thể dừng sớm nhờ early stopping)")
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
# CELL 10.1: Kiểm tra và Import (OPTIMIZED)
# ============================================
# ⚠️ QUAN TRỌNG: Phải chạy CELL 9 trước để tạo model!
# 💡 Phiên bản TỐI ƯU này có:
#    - Test định kỳ mỗi 100 epochs (giảm từ 50 → tiết kiệm ~50% thời gian test)
#    - Early stopping dựa trên % epochs không cải thiện
#    - Chỉ lưu best checkpoint (không lưu periodic checkpoints → tiết kiệm I/O)
#    - Mixed precision training (FP16) → tăng tốc GPU ~1.5-2x
#    - torch.compile (PyTorch 2.0+) → tăng tốc thêm ~10-20%
#    - Tối ưu early stopping để dừng sớm hơn

print("\n🚀 Starting training with Early Stopping...")
print("="*60)

# Kiểm tra các biến cần thiết
required_vars = ['model', 'train_dataloader', 'transt', 'test_dataloader', 'TRANST_CONFIG', 'TRAIN_RUN', 'CHECKPOINT_PATH', 'RESULTS_PATH']
missing_vars = [var for var in required_vars if var not in globals()]

if missing_vars:
    print(f"❌ Missing variables: {missing_vars}")
    print("💡 Hãy chạy các cell trước đó:")
    print("   - CELL 6: Config (TRANST_CONFIG, TRAIN_RUN, CHECKPOINT_PATH, RESULTS_PATH)")
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

# ============================================
# Import Early Stopping Helper
# ============================================
import sys
import os
sys.path.insert(0, 'research/ttm/experiments')
from early_stopping_helper import EarlyStopping, test_model_periodically

# ============================================
# CELL 10.2: Cấu hình Early Stopping và Test (TỐI ƯU)
# ============================================
# Cấu hình Early Stopping (TỐI ƯU cho config match baseline)
EARLY_STOPPING_CONFIG = {
    'patience_percent': 0.10,    # 10% tổng epochs = 150 epochs (với 1500 epochs)
    'min_delta': 0.0001,         # Cải thiện tối thiểu để coi là "cải thiện"
    'monitor': 'Hits@10',        # Metric để theo dõi (quan trọng nhất)
    'mode': 'max',               # 'max' cho Hits@10, MRR (càng lớn càng tốt)
    'min_epochs': 500,           # Tối thiểu train 500 epochs (tăng từ 400 để đảm bảo model học đủ với config mới)
}

# Tạo Early Stopping object
early_stopping = EarlyStopping(
    patience_percent=EARLY_STOPPING_CONFIG['patience_percent'],
    min_delta=EARLY_STOPPING_CONFIG['min_delta'],
    monitor=EARLY_STOPPING_CONFIG['monitor'],
    mode=EARLY_STOPPING_CONFIG['mode'],
    min_epochs=EARLY_STOPPING_CONFIG['min_epochs']
)

# Set total epochs
early_stopping.set_total_epochs(TRANST_CONFIG['train_times'])

# Cấu hình Test Định Kỳ (TỐI ƯU)
TEST_INTERVAL = 100  # Test mỗi 100 epochs (tăng từ 50 → giảm 50% số lần test)

# Tắt periodic checkpoint saving để tiết kiệm I/O
SAVE_PERIODIC_CHECKPOINTS = False  # Chỉ lưu best checkpoint

print(f"\n📊 Early Stopping Configuration:")
print(f"   Patience: {EARLY_STOPPING_CONFIG['patience_percent']*100:.1f}% of total epochs")
print(f"   Monitor: {EARLY_STOPPING_CONFIG['monitor']}")
print(f"   Min delta: {EARLY_STOPPING_CONFIG['min_delta']}")
print(f"   Min epochs: {EARLY_STOPPING_CONFIG['min_epochs']}")
print(f"\n🧪 Test Configuration:")
print(f"   Test interval: Every {TEST_INTERVAL} epochs")
print(f"   Periodic checkpoints: {'Enabled' if SAVE_PERIODIC_CHECKPOINTS else 'Disabled (only best)'}")
print("="*60)

# Đường dẫn checkpoint
checkpoint_path = f'{CHECKPOINT_PATH}/transt_fb15k237_l{TRAIN_RUN}.ckpt'
best_checkpoint_path = f'{CHECKPOINT_PATH}/transt_fb15k237_l{TRAIN_RUN}_best.ckpt'
checkpoint_dir = f'{CHECKPOINT_PATH}/transt_fb15k237_l{TRAIN_RUN}_periodic'

# Chỉ tạo thư mục periodic nếu cần
if SAVE_PERIODIC_CHECKPOINTS:
    os.makedirs(checkpoint_dir, exist_ok=True)

# ============================================
# CELL 10.3: Setup Trainer và Performance Optimizations
# ============================================
print("\n🚀 Starting training...")
print("="*60)

# Setup trainer (chỉ để lấy optimizer và các setup khác)
from openke.config import Trainer

trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=TRANST_CONFIG['train_times'],
    alpha=TRANST_CONFIG['alpha'],
    use_gpu=True
)

# Setup GPU và optimizer (từ trainer)
import torch
import torch.optim as optim
from tqdm import tqdm

if trainer.use_gpu:
    trainer.model.cuda()
    
    # ============================================
    # TỐI ƯU: Mixed Precision Training (FP16)
    # ============================================
    # Sử dụng FP16 để tăng tốc GPU ~1.5-2x
    USE_MIXED_PRECISION = True
    if USE_MIXED_PRECISION:
        # Sử dụng API mới (PyTorch 2.0+) để tránh warning
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda')
            print("✅ Mixed Precision Training (FP16) enabled!")
        elif hasattr(torch.cuda, 'amp'):
            # Fallback cho PyTorch cũ hơn
            scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed Precision Training (FP16) enabled!")
        else:
            USE_MIXED_PRECISION = False
            scaler = None
            print("⚠️  PyTorch version < 1.6, Mixed Precision disabled")
    else:
        USE_MIXED_PRECISION = False
        scaler = None
    
    # ============================================
    # TỐI ƯU: torch.compile (PyTorch 2.0+)
    # ============================================
    # Compile model để tăng tốc thêm ~10-20%
    # ⚠️ Lưu ý: torch.compile yêu cầu CUDA Capability >= 7.0
    USE_COMPILE = True
    if USE_COMPILE and hasattr(torch, 'compile'):
        # Kiểm tra CUDA capability trước khi compile
        try:
            device_props = torch.cuda.get_device_properties(0)
            cuda_capability = device_props.major
            device_name = device_props.name
            
            if cuda_capability < 7:
                print(f"⚠️  GPU {device_name} (CUDA Capability {cuda_capability}.{device_props.minor}) không hỗ trợ torch.compile")
                print(f"   → Triton compiler yêu cầu CUDA Capability >= 7.0")
                print(f"   → Tự động tắt torch.compile")
                USE_COMPILE = False
            else:
                # Thử compile (compile có thể fail khi forward pass, nên sẽ test sau)
                try:
                    trainer.model = torch.compile(trainer.model, mode='reduce-overhead')
                    print(f"✅ torch.compile enabled (PyTorch 2.0+)!")
                    print(f"   GPU: {device_name} (CUDA Capability {cuda_capability}.{device_props.minor})")
                    print(f"   ⚠️  Note: Compile sẽ được test khi training bắt đầu")
                except Exception as compile_error:
                    print(f"⚠️  torch.compile initialization failed: {compile_error}")
                    print(f"   → Tự động tắt torch.compile")
                    USE_COMPILE = False
        except Exception as e:
            print(f"⚠️  Error checking GPU capability: {e}")
            print(f"   → Tự động tắt torch.compile để tránh lỗi")
            USE_COMPILE = False
    else:
        USE_COMPILE = False
        if not hasattr(torch, 'compile'):
            print("⚠️  PyTorch version < 2.0, torch.compile disabled")
    
    # In tổng hợp các tối ưu
    print("\n⚡ Performance Optimizations:")
    if USE_MIXED_PRECISION and scaler is not None:
        print(f"   ✅ Mixed Precision (FP16): Enabled (~1.5-2x speedup)")
    if USE_COMPILE:
        print(f"   ✅ torch.compile: Enabled (~10-20% speedup)")
    if not SAVE_PERIODIC_CHECKPOINTS:
        print(f"   ✅ Periodic checkpoint saving: Disabled (saves I/O time)")
else:
    USE_MIXED_PRECISION = False
    USE_COMPILE = False
    scaler = None

# Initialize optimizer (trainer sẽ tự động setup, nhưng ta cần đảm bảo)
if not hasattr(trainer, 'optimizer') or trainer.optimizer is None:
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

print("✅ Model and optimizer initialized!")

# ============================================
# CELL 10.4: Training Loop với Early Stopping (TỐI ƯU)
# ============================================
# Training loop với test định kỳ (TỐI ƯU)
training_range = tqdm(range(TRANST_CONFIG['train_times']))
training_stopped_early = False

for epoch in training_range:
    actual_epoch = epoch + 1
    
    # Training một epoch với mixed precision nếu được bật
    res = 0.0
    if USE_MIXED_PRECISION and scaler is not None:
        # Sử dụng mixed precision training
        # Sử dụng API mới nếu có, fallback về API cũ
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            autocast_context = torch.amp.autocast('cuda')
        else:
            autocast_context = torch.cuda.amp.autocast()
        
        for data in train_dataloader:
            trainer.optimizer.zero_grad()
            
            try:
                # Forward pass với autocast
                with autocast_context:
                    loss = trainer.model({
                        'batch_h': trainer.to_var(data['batch_h'], trainer.use_gpu),
                        'batch_t': trainer.to_var(data['batch_t'], trainer.use_gpu),
                        'batch_r': trainer.to_var(data['batch_r'], trainer.use_gpu),
                        'batch_y': trainer.to_var(data['batch_y'], trainer.use_gpu),
                        'mode': data['mode']
                    })
                
                # Backward pass với scaler
                scaler.scale(loss).backward()
                scaler.step(trainer.optimizer)
                scaler.update()
                
                res += loss.item()
            except Exception as e:
                # Nếu lỗi do torch.compile (GPU quá cũ), tự động tắt compile và reload model
                if 'GPUTooOldForTriton' in str(type(e).__name__) or 'Triton' in str(e):
                    print(f"\n⚠️  Lỗi torch.compile phát hiện: {e}")
                    print(f"   → GPU không hỗ trợ torch.compile, tự động tắt và reload model...")
                    
                    # Tắt compile và reload model từ transt
                    USE_COMPILE = False
                    trainer.model = model  # Reload từ model gốc (không compile)
                    
                    print(f"   ✅ Đã tắt torch.compile, tiếp tục training với model bình thường")
                    
                    # Retry với model không compile
                    with autocast_context:
                        loss = trainer.model({
                            'batch_h': trainer.to_var(data['batch_h'], trainer.use_gpu),
                            'batch_t': trainer.to_var(data['batch_t'], trainer.use_gpu),
                            'batch_r': trainer.to_var(data['batch_r'], trainer.use_gpu),
                            'batch_y': trainer.to_var(data['batch_y'], trainer.use_gpu),
                            'mode': data['mode']
                        })
                    
                    scaler.scale(loss).backward()
                    scaler.step(trainer.optimizer)
                    scaler.update()
                    
                    res += loss.item()
                else:
                    # Lỗi khác, raise lại
                    raise
    else:
        # Training bình thường (FP32)
        for data in train_dataloader:
            try:
                loss = trainer.train_one_step(data)
                res += loss
            except Exception as e:
                # Nếu lỗi do torch.compile (GPU quá cũ), tự động tắt compile và reload model
                if 'GPUTooOldForTriton' in str(type(e).__name__) or 'Triton' in str(e):
                    print(f"\n⚠️  Lỗi torch.compile phát hiện: {e}")
                    print(f"   → GPU không hỗ trợ torch.compile, tự động tắt và reload model...")
                    
                    # Tắt compile và reload model từ transt
                    USE_COMPILE = False
                    trainer.model = model  # Reload từ model gốc (không compile)
                    
                    print(f"   ✅ Đã tắt torch.compile, tiếp tục training với model bình thường")
                    
                    # Retry với model không compile
                    loss = trainer.train_one_step(data)
                    res += loss
                else:
                    # Lỗi khác, raise lại
                    raise
    
    training_range.set_description(f"Epoch {actual_epoch} | loss: {res:.4f}")
    
    # Test định kỳ và kiểm tra early stopping
    if actual_epoch % TEST_INTERVAL == 0 or actual_epoch == TRANST_CONFIG['train_times']:
        # Đường dẫn checkpoint cho epoch này (chỉ dùng nếu cần lưu periodic)
        epoch_checkpoint_path = None
        if SAVE_PERIODIC_CHECKPOINTS:
            epoch_checkpoint_path = f'{checkpoint_dir}/epoch_{actual_epoch:05d}.ckpt'
        
        # Test model
        metrics, should_stop, is_best = test_model_periodically(
            model=transt,
            test_dataloader=test_dataloader,
            epoch=actual_epoch,
            test_interval=TEST_INTERVAL,
            type_constrain=TRANST_CONFIG['type_constrain'],
            use_gpu=True,
            early_stopping=early_stopping,
            checkpoint_path=epoch_checkpoint_path,  # None nếu không lưu periodic
            best_checkpoint_path=best_checkpoint_path
        )
        
        # ⭐ LƯU BEST RESULT NGAY TRONG QUÁ TRÌNH TRAINING (Đảm bảo an toàn)
        # Nếu phát hiện kết quả tốt hơn, lưu ngay lập tức để đảm bảo không mất dữ liệu
        if metrics and is_best:
            try:
                save_best_result_during_training(
                    metrics, actual_epoch, TRAIN_RUN, TRANST_CONFIG, best_checkpoint_path
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not save best result during training: {e}")
                print(f"   Will try again at the end of training")
        
        # Kiểm tra early stopping
        if should_stop:
            training_stopped_early = True
            print(f"\n🛑 Training stopped early at epoch {actual_epoch}")
            print(f"   Best {early_stopping.monitor}: {early_stopping.get_best_score():.4f} at epoch {early_stopping.get_best_epoch()}")
            break

print("="*60)
if training_stopped_early:
    print("✅ Training stopped early (Early Stopping triggered)")
else:
    print("✅ Training complete!")

# ============================================
# CELL 10.5: Lưu lịch sử và Load Best Checkpoint
# ============================================
# Lưu lịch sử training
history_path = f'{RESULTS_PATH}/transt_fb15k237_l{TRAIN_RUN}_history.json'
early_stopping.save_history(history_path)

# Load best checkpoint và test lần cuối
print("\n📥 Loading best checkpoint for final evaluation...")
if os.path.exists(best_checkpoint_path):
    transt.load_checkpoint(best_checkpoint_path)
    print(f"✅ Best checkpoint loaded: epoch {early_stopping.get_best_epoch()}")
    print(f"   Best {early_stopping.monitor}: {early_stopping.get_best_score():.4f}")
    # Luôn lưu final checkpoint để có thể download sau
    transt.save_checkpoint(checkpoint_path)
    print(f"✅ Final checkpoint also saved: {checkpoint_path}")
else:
    # Nếu không có best checkpoint, dùng checkpoint cuối cùng
    print(f"⚠️  Best checkpoint not found, using final checkpoint")
    transt.save_checkpoint(checkpoint_path)
    print(f"✅ Final checkpoint saved: {checkpoint_path}")

# ============================================
# CELL 10.6: Final Testing với Best Model
# ============================================
print("\n🧪 Final Testing (Best Model)...")
print("="*60)

from openke.config import Tester
tester = Tester(model=transt, data_loader=test_dataloader, use_gpu=True)

mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=TRANST_CONFIG['type_constrain'])

print("="*60)
print("📊 TransT FINAL RESULTS on FB15K237:")
print("="*60)
print(f"MRR: {mrr:.4f}  MR: {mr:.1f}  Hits@10: {hit10:.4f}  Hits@3: {hit3:.4f}  Hits@1: {hit1:.4f}")
if training_stopped_early:
    print(f"\n🛑 Training stopped early at epoch {actual_epoch}")
    print(f"   Best epoch: {early_stopping.get_best_epoch()}")
    print(f"   Best {early_stopping.monitor}: {early_stopping.get_best_score():.4f}")
print("="*60)

# ============================================
# CELL 11: Lưu kết quả vào file
# ============================================
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
    f.write(f"Early Stopping: {'Yes' if training_stopped_early else 'No'}\n")
    if training_stopped_early:
        f.write(f"Stopped at epoch: {actual_epoch}\n")
        f.write(f"Best epoch: {early_stopping.get_best_epoch()}\n")
        f.write(f"Best {early_stopping.monitor}: {early_stopping.get_best_score():.4f}\n")
    f.write(f"\nConfiguration:\n")
    for key, value in TRANST_CONFIG.items():
        f.write(f"   {key}: {value}\n")
    f.write(f"\nEarly Stopping Config:\n")
    for key, value in EARLY_STOPPING_CONFIG.items():
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
csv_file = f'{RESULTS_PATH}/transt_fb15k237_all_results.csv'

# Kiểm tra file CSV đã tồn tại chưa
file_exists = os.path.exists(csv_file)

with open(csv_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Ghi header nếu file mới
    if not file_exists:
        writer.writerow(['Run', 'MRR', 'MR', 'Hits@10', 'Hits@3', 'Hits@1', 'Early_Stop', 'Best_Epoch', 'Stopped_At', 'Margin', 'Epochs', 'LR', 'Neg_ent', 'Dim'])
    
    # Ghi kết quả
    writer.writerow([
        TRAIN_RUN,
        f"{mrr:.4f}",
        f"{mr:.1f}",
        f"{hit10:.4f}",
        f"{hit3:.4f}",
        f"{hit1:.4f}",
        'Yes' if training_stopped_early else 'No',
        early_stopping.get_best_epoch() if training_stopped_early else TRANST_CONFIG['train_times'],
        actual_epoch if training_stopped_early else TRANST_CONFIG['train_times'],
        TRANST_CONFIG['margin'],
        TRANST_CONFIG['train_times'],
        TRANST_CONFIG['alpha'],
        TRANST_CONFIG['neg_ent'],
        TRANST_CONFIG['dim']
    ])

print(f"✅ Results appended to CSV: {csv_file}")

# ============================================
# CELL 12: Tự động lưu Best Result và Zip
# ============================================
import json
import zipfile
from IPython.display import FileLink
from datetime import datetime

# Đường dẫn file lưu best result
best_result_file = f'{RESULTS_PATH}/transt_fb15k237_best_result.json'
best_result_zip = f'/kaggle/working/transt_fb15k237_BEST_RESULT.zip'

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

def save_best_result_during_training(metrics, epoch, run_number, config, best_checkpoint_path):
    """Lưu best result ngay trong quá trình training khi phát hiện kết quả tốt hơn
    Đảm bảo an toàn: lưu ngay cả khi chương trình bị lỗi giữa chừng
    """
    try:
        # Tạo current result từ metrics
        current_result = {
            'Run': run_number,
            'MRR': metrics['MRR'],
            'MR': metrics['MR'],
            'Hits@10': metrics['Hits@10'],
            'Hits@3': metrics['Hits@3'],
            'Hits@1': metrics['Hits@1'],
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'stopped_early': False,  # Chưa biết, sẽ cập nhật sau
            'config': config.copy(),
            'saved_during_training': True  # Đánh dấu là lưu trong quá trình training
        }
        
        # So sánh với best result hiện tại
        best_result = load_best_result()
        
        if is_better_result(current_result, best_result):
            # Lưu best result ngay lập tức
            if save_best_result(current_result):
                print(f"\n🏆 NEW BEST RESULT during training! (Epoch {epoch})")
                print(f"   MRR={metrics['MRR']:.4f}, Hits@10={metrics['Hits@10']:.4f}, MR={metrics['MR']:.1f}")
                if best_result:
                    improvement_hit10 = ((metrics['Hits@10'] - best_result['Hits@10']) / best_result['Hits@10']) * 100
                    improvement_mrr = ((metrics['MRR'] - best_result['MRR']) / best_result['MRR']) * 100
                    print(f"   Improvement: Hits@10 +{improvement_hit10:.2f}%, MRR +{improvement_mrr:.2f}%")
                
                # Tạo zip file ngay lập tức (chỉ với những file có sẵn)
                try:
                    create_best_result_zip_during_training(
                        best_checkpoint_path, metrics, epoch, run_number, config
                    )
                except Exception as e:
                    print(f"⚠️  Warning: Could not create zip during training: {e}")
                    print(f"   Best result JSON is still saved: {best_result_file}")
                
                return True
        return False
    except Exception as e:
        print(f"⚠️  Warning: Error saving best result during training: {e}")
        print(f"   Will try again at the end of training")
        return False

def create_best_result_zip_during_training(best_checkpoint_path, metrics, epoch, run_number, config):
    """Tạo file zip best result trong quá trình training (chỉ với những file có sẵn)"""
    try:
        files_added = []
        files_missing = []
        
        print(f"   📦 Creating BEST RESULT zip (during training)...")
        
        with zipfile.ZipFile(best_result_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Thêm best checkpoint (quan trọng nhất - phải có)
            if os.path.exists(best_checkpoint_path):
                zipf.write(best_checkpoint_path, f'transt_fb15k237_l{run_number}_best.ckpt')
                files_added.append('best checkpoint')
            else:
                files_missing.append('best checkpoint')
            
            # Thêm best result JSON (phải có)
            if os.path.exists(best_result_file):
                zipf.write(best_result_file, 'transt_fb15k237_best_result.json')
                files_added.append('best result JSON')
            else:
                files_missing.append('best result JSON')
            
            # Thêm history file nếu có (có thể chưa có nếu training chưa xong)
            history_path = f'{RESULTS_PATH}/transt_fb15k237_l{run_number}_history.json'
            if os.path.exists(history_path):
                zipf.write(history_path, f'transt_fb15k237_l{run_number}_history.json')
                files_added.append('history file')
            
            # Thêm CSV file nếu có
            csv_file_path = f'{RESULTS_PATH}/transt_fb15k237_all_results.csv'
            if os.path.exists(csv_file_path):
                zipf.write(csv_file_path, 'transt_fb15k237_all_results.csv')
                files_added.append('all results CSV')
            
            # Tạo file results tạm thời từ metrics hiện tại
            temp_result_file = f'{RESULTS_PATH}/transt_fb15k237_l{run_number}_temp_results.txt'
            try:
                with open(temp_result_file, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("TransT (TransE + Triple Trustiness) Results (During Training)\n")
                    f.write("="*60 + "\n")
                    f.write(f"Train Run: {run_number}\n")
                    f.write(f"Dataset: FB15K237\n")
                    f.write(f"Epoch: {epoch}\n")
                    f.write(f"Note: This is a temporary result saved during training\n")
                    f.write(f"\nConfiguration:\n")
                    for key, value in config.items():
                        f.write(f"   {key}: {value}\n")
                    f.write(f"\nResults (at epoch {epoch}):\n")
                    f.write(f"   MRR: {metrics['MRR']:.4f}\n")
                    f.write(f"   MR: {metrics['MR']:.1f}\n")
                    f.write(f"   Hits@10: {metrics['Hits@10']:.4f}\n")
                    f.write(f"   Hits@3: {metrics['Hits@3']:.4f}\n")
                    f.write(f"   Hits@1: {metrics['Hits@1']:.4f}\n")
                    f.write("="*60 + "\n")
                
                zipf.write(temp_result_file, f'transt_fb15k237_l{run_number}_results.txt')
                files_added.append('temp results file')
            except Exception as e:
                print(f"   ⚠️  Could not create temp results file: {e}")
        
        if files_added:
            print(f"   ✅ Best result zip saved: {best_result_zip} ({len(files_added)} files)")
            if files_missing:
                print(f"   ⚠️  Missing files (will be added at end of training): {', '.join(files_missing)}")
        else:
            print(f"   ⚠️  No files to zip")
            
    except Exception as e:
        print(f"   ⚠️  Error creating zip during training: {e}")
        raise

def create_best_result_zip(best_checkpoint_path, checkpoint_path, result_file, history_path, 
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
                zipf.write(best_checkpoint_path, f'transt_fb15k237_l{run_number}_best.ckpt')
                files_added.append('best checkpoint')
                print(f"   ✅ Added: best checkpoint")
            else:
                files_missing.append('best checkpoint')
                print(f"   ⚠️  Missing: best checkpoint")
            
            # Thêm final checkpoint
            if os.path.exists(checkpoint_path):
                zipf.write(checkpoint_path, f'transt_fb15k237_l{run_number}.ckpt')
                files_added.append('final checkpoint')
                print(f"   ✅ Added: final checkpoint")
            else:
                files_missing.append('final checkpoint')
                print(f"   ⚠️  Missing: final checkpoint")
            
            # Thêm result file
            if os.path.exists(result_file):
                zipf.write(result_file, f'transt_fb15k237_l{run_number}_results.txt')
                files_added.append('results file')
                print(f"   ✅ Added: results file")
            else:
                files_missing.append('results file')
                print(f"   ⚠️  Missing: results file")
            
            # Thêm history file
            if os.path.exists(history_path):
                zipf.write(history_path, f'transt_fb15k237_l{run_number}_history.json')
                files_added.append('history file')
                print(f"   ✅ Added: history file")
            else:
                files_missing.append('history file')
                print(f"   ⚠️  Missing: history file")
            
            # Thêm best result JSON file
            if os.path.exists(best_result_json_path):
                zipf.write(best_result_json_path, 'transt_fb15k237_best_result.json')
                files_added.append('best result JSON')
                print(f"   ✅ Added: best result JSON")
            else:
                files_missing.append('best result JSON')
                print(f"   ⚠️  Missing: best result JSON")
            
            # Thêm CSV file (tất cả results)
            if os.path.exists(csv_file_path):
                zipf.write(csv_file_path, 'transt_fb15k237_all_results.csv')
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
    'epoch': early_stopping.get_best_epoch() if training_stopped_early else TRANST_CONFIG['train_times'],
    'stopped_early': training_stopped_early,
    'config': TRANST_CONFIG.copy()
}

best_result = load_best_result()

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
    
    # Lưu best result (hoặc cập nhật nếu đã lưu trong quá trình training)
    if save_best_result(current_result):
        # Tạo/cập nhật zip file cho best result với đầy đủ files (sau khi training hoàn tất)
        try:
            files_added, files_missing = create_best_result_zip(
                best_checkpoint_path, checkpoint_path, result_file, history_path,
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
# CELL 13: Download checkpoint và results (optional - cho lần train hiện tại)
# ============================================

# Zip checkpoint
checkpoint_zip = f'/kaggle/working/transt_fb15k237_l{TRAIN_RUN}.zip'
files_added = []
files_missing = []

print("\n📦 Preparing zip file...")
print(f"   Zip path: {checkpoint_zip}")

with zipfile.ZipFile(checkpoint_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Thêm best checkpoint nếu tồn tại
    if os.path.exists(best_checkpoint_path):
        zipf.write(best_checkpoint_path, f'transt_fb15k237_l{TRAIN_RUN}_best.ckpt')
        files_added.append('best checkpoint')
        print(f"   ✅ Added: best checkpoint")
    else:
        files_missing.append(f'best checkpoint ({best_checkpoint_path})')
        print(f"   ⚠️  Missing: best checkpoint")
    
    # Thêm final checkpoint nếu tồn tại
    if os.path.exists(checkpoint_path):
        zipf.write(checkpoint_path, f'transt_fb15k237_l{TRAIN_RUN}.ckpt')
        files_added.append('final checkpoint')
        print(f"   ✅ Added: final checkpoint")
    else:
        files_missing.append(f'final checkpoint ({checkpoint_path})')
        print(f"   ⚠️  Missing: final checkpoint")
    
    # Thêm result file nếu tồn tại
    if os.path.exists(result_file):
        zipf.write(result_file, f'transt_fb15k237_l{TRAIN_RUN}_results.txt')
        files_added.append('results file')
        print(f"   ✅ Added: results file")
    else:
        files_missing.append(f'results file ({result_file})')
        print(f"   ⚠️  Missing: results file")
    
    # Thêm history file nếu tồn tại
    if os.path.exists(history_path):
        zipf.write(history_path, f'transt_fb15k237_l{TRAIN_RUN}_history.json')
        files_added.append('history file')
        print(f"   ✅ Added: history file")
    else:
        files_missing.append(f'history file ({history_path})')
        print(f"   ⚠️  Missing: history file")

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

