# ============================================
# CELL 1: Clone OpenKE
# ============================================
# 💡 Nếu Kaggle không có internet, có thể load OpenKE từ Kaggle Dataset
#    Thêm dataset chứa OpenKE và uncomment phần code bên dưới

# CÁCH 1: Clone từ GitHub (cần internet)
!git clone https://github.com/thunlp/OpenKE.git
!cd OpenKE && git checkout OpenKE-PyTorch

# CÁCH 2: Load từ Kaggle Dataset (nếu không có internet)
# Uncomment nếu cần:
# import shutil
# import os
# OPENKE_DATASET_NAME = "openke-pytorch"  # ⬅️ Sửa tên dataset nếu khác
# openke_dataset_path = f'/kaggle/input/{OPENKE_DATASET_NAME}'
# if os.path.exists(openke_dataset_path):
#     print("📂 Copying OpenKE from Kaggle Dataset...")
#     shutil.copytree(openke_dataset_path, 'OpenKE', dirs_exist_ok=True)
#     print("✅ OpenKE copied from dataset!")
# else:
#     print(f"⚠️  OpenKE dataset not found: {openke_dataset_path}")
#     print("💡 Will try to clone from GitHub...")

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

# ============================================
# CONFIG VERSION 1: Dựa trên kết quả tốt nhất (Lần 1)
# ============================================
# 📊 Config này dựa trên kết quả lần 1 (tốt nhất):
# - MRR: 0.2528, MR: 193.6, Hits@10: 0.4143
# - Config: dim=200, margin=5.0, alpha=1.0, train_times=1000
# - use_cross_entropy=True, alpha_trust=0.5, beta_trust=0.5
# - type_constrain=False
#
# 💡 Khuyến nghị: Tăng train_times lên 1500 để match baseline
#    (có thể cải thiện kết quả thêm)

TRANST_CONFIG = {
    'dim': 200,              # Embedding dimension (giữ như lần 1)
    'margin': 5.0,           # Margin for loss (giữ như lần 1)
    'train_times': 1500,     # Number of epochs (tăng từ 1000 → 1500 để match baseline)
    'alpha': 1.0,            # Learning rate (QUAN TRỌNG: giữ 1.0 như lần 1)
    'nbatches': 100,         # Number of batches per epoch (giữ như lần 1)
    'neg_ent': 10,           # Negative samples per positive (giữ như lần 1)
    'threads': 0,            # Number of threads (0 = single thread)
    'p_norm': 1,             # L1 distance
    'norm_flag': True,       # Normalize embeddings
    'type_constrain': False, # Use type constraint in evaluation (giữ False như lần 1)
    
    # TTM specific - Config của lần 1 (tốt nhất)
    'use_trustiness': True,  # Có dùng trustiness không
    'alpha_trust': 0.5,      # Weight cho type-based trustiness (QUAN TRỌNG: giữ 0.5 như lần 1)
    'beta_trust': 0.5,       # Weight cho description-based trustiness (QUAN TRỌNG: giữ 0.5 như lần 1)
    'use_cross_entropy': True,  # QUAN TRỌNG: Dùng cross-entropy loss (True như lần 1)
}

print("\n" + "="*60)
print("⚙️  TransT (TransE + Triple Trustiness) Configuration")
print("="*60)
print(f"📊 Train Run: {TRAIN_RUN}")
print(f"📂 Dataset: FB15K237")
print(f"📋 Config: Version 1 (Based on best results - Run 1)")
print("\n📋 Hyperparameters:")
for key, value in TRANST_CONFIG.items():
    print(f"   {key:20s}: {value}")

# Tính toán thời gian ước tính
total_iterations = TRANST_CONFIG['train_times'] * TRANST_CONFIG['nbatches']
print("\n⏱️  Thông tin training:")
print(f"   Tổng iterations: {total_iterations:,} ({TRANST_CONFIG['train_times']} epochs × {TRANST_CONFIG['nbatches']} batches)")
print(f"   ⚠️  Lưu ý: Thời gian thực tế có thể khác với ước tính ban đầu")
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
# ⚠️ QUAN TRỌNG: Phải chạy CELL 9 trước để tạo model!
# 💡 Phiên bản này KHÔNG có checkpoint định kỳ (đơn giản hơn)
# 💡 Test sẽ tự động chạy ngay sau khi train xong

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
print("="*60)

# Trainer đơn giản (không có checkpoint định kỳ)
trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=TRANST_CONFIG['train_times'],
    alpha=TRANST_CONFIG['alpha'],
    use_gpu=True
)

trainer.run()

print("="*60)
print("✅ Training complete!")

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
# CELL 12: Download checkpoint và results (optional)
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

