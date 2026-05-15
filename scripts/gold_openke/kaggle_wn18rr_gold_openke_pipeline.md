# Kaggle — WN18RR + GOLD + TransE (OpenKE)

Copy từng khối **Cell N — code** vào notebook GPU trên Kaggle (giống [`kaggle_gold_openke_pipeline.md`](kaggle_gold_openke_pipeline.md), nhưng benchmark **WN18RR**). **Accelerator:** khuyến nghị **GPU T4** (torch CUDA mặc định trên Kaggle; phù hợp GOLD + sentence-transformers). Không dùng wheel CPU-only trên Kaggle.

**Gắn Input (khuyến nghị):** dataset chứa `OpenKE-OpenKE-PyTorch` (có `scripts/gold_openke/openke_to_gold.py`) + `GOLD-main` + (tuỳ chọn) thư mục `WN18RR` hoặc `benchmarks/WN18RR` đủ `entity2id.txt` … Hoặc bật **Internet** và để `GIT_OPENKE` / `GIT_GOLD` + clone upstream lấy `benchmarks/WN18RR` (cell 2).

**Lưu ý WN18RR:** thường `TRANSE_TYPE_CONSTRAIN = False`; dim/margin/alpha nhỏ hơn FB15K237 (xem cell 1).

---

## Cell 1 — code

```python
import os
import shutil
import subprocess
import sys

ON_KAGGLE = os.path.isdir("/kaggle/working")
_cwd = os.getcwd()
if ON_KAGGLE:
    WORK = "/kaggle/working"
elif os.path.isfile(os.path.join(_cwd, "openke_to_gold.py")):
    WORK = os.path.abspath(os.path.join(_cwd, "..", ".."))
else:
    WORK = os.path.abspath(_cwd)

# ===== Hyper-parameters (chỉ chỉnh ở đây) =====
# WN18RR nhỏ hơn FB15K237: dim ~100, alpha ~0.1, margin ~5.5–6, neg_ent ~5.
# Nếu sát giờ session: giảm TRAIN_TIMES / TRANSE_NBATCHES hoặc GOLD_BATCH_SIZE.
# GOLD
EPOCH = 1
GOLD_BATCH_SIZE = 128  # 96 / 64 nếu OOM
GOLD_PTLM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GOLD_LR = 0.001
GOLD_LOCAL_LAMBDA = 0.1
GOLD_GLOBAL_LAMBDA = 0.01
GOLD_NEG_CNT = 1
GOLD_SEED = 5

# TransE (OpenKE) — WN18RR
TRAIN_TIMES = 2000
TRANSE_DIM = 100
TRANSE_NBATCHES = 50
TRANSE_NEG_ENT = 5
TRANSE_MARGIN = 6.0
TRANSE_ALPHA = 0.1
TRANSE_THREADS = 0
TRANSE_BERN_FLAG = 1
TRANSE_FILTER_FLAG = 1
TRANSE_TYPE_CONSTRAIN = False

# TransE — checkpoint định kỳ & resume (khi session Kaggle bị ngắt)
# Lưu mỗi TRANSE_SAVE_EVERY epoch vào OPENKE_ROOT/checkpoint_gold_clean_wn18rr/
TRANSE_SAVE_EVERY = 100  # 0 = tắt lưu định kỳ (vẫn lưu file cuối transe_gold_clean.ckpt)
TRANSE_CHECKPOINT_PREFIX = "transe_gold_clean_wn18rr"
TRANSE_RESUME_LATEST = False  # True = tiếp tục từ file *_epoch_*.ckpt mới nhất trong thư mục trên
TRANSE_RESUME_CKPT = None  # hoặc đường dẫn đầy đủ tới .ckpt (ưu tiên hơn TRANSE_RESUME_LATEST)

INPUT_BUNDLE = None

GIT_OPENKE = "https://github.com/Congvinh2004/ML_KGE.git"
GIT_GOLD = "https://github.com/Congvinh2004/ML_GOLD-main.git"

OPENKE_ROOT = os.path.join(WORK, "OpenKE-OpenKE-PyTorch")
GOLD_ROOT = os.path.join(WORK, "GOLD-main")
OPENKE_DATA = os.path.join(WORK, "data", "WN18RR")

DATASET_TAG = "WN18RR_openke"
GOLD_LAYOUT = os.path.join(WORK, "gold_data", DATASET_TAG)
MODEL_NAME = "kaggle_wn18rr_gold1"  # TSV: WN18RR_openke_kaggle_wn18rr_gold1.tsv
RULE_TOP_K = 100

os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.dirname(OPENKE_DATA), exist_ok=True)

print("WORK:", WORK)
print("ON_KAGGLE:", ON_KAGGLE)
```

## Cell 2 — code

```python
def run(cmd, cwd=None):
    print("$", " ".join(cmd) if isinstance(cmd, list) else cmd)
    if isinstance(cmd, str):
        r = subprocess.run(cmd, shell=True, cwd=cwd)
    else:
        r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("Command failed")


import re


def parse_gold_stdout(text: str):
    """Lấy Acc/AUC từ stdout/stderr của gold.py ([ACC REPORT] / [TEST RESULT])."""
    valid_acc = valid_auc = test_acc = test_auc = None
    for m in re.finditer(
        r"\[ACC REPORT\][^\n]*Acc = ([0-9.]+),\s*AUC = ([0-9.]+)",
        text,
    ):
        valid_acc, valid_auc = float(m.group(1)), float(m.group(2))
    for m in re.finditer(
        r"\[TEST RESULT\][^\n]*Acc = ([0-9.]+),\s*AUC = ([0-9.]+)",
        text,
    ):
        test_acc, test_auc = float(m.group(1)), float(m.group(2))
    return {
        "gold_valid_acc": valid_acc,
        "gold_valid_auc": valid_auc,
        "gold_test_acc": test_acc,
        "gold_test_auc": test_auc,
    }


def _kaggle_dataset_dirs():
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return []
    return [
        os.path.join(base, d)
        for d in sorted(os.listdir(base))
        if os.path.isdir(os.path.join(base, d))
    ]


def _find_openke_repo_root(search_roots):
    """Thư mục chứa scripts/gold_openke/openke_to_gold.py."""
    subnames = ("OpenKE-OpenKE-PyTorch", "OpenKE-PyTorch")
    for root in search_roots:
        for sub in subnames:
            p = os.path.join(root, sub)
            if os.path.isfile(os.path.join(p, "scripts", "gold_openke", "openke_to_gold.py")):
                return p
        if os.path.isfile(os.path.join(root, "scripts", "gold_openke", "openke_to_gold.py")):
            return root
    return None


def _find_gold_root(search_roots):
    for root in search_roots:
        p = os.path.join(root, "GOLD-main")
        if os.path.isfile(os.path.join(p, "gold.py")):
            return p
        if os.path.isfile(os.path.join(root, "gold.py")):
            return root
    return None


def _find_wn18rr_data_dir(search_roots):
    """Thư mục benchmark WN18RR (entity2id.txt)."""
    rel_paths = (
        os.path.join("WN18RR"),
        os.path.join("benchmarks", "WN18RR"),
        os.path.join("OpenKE-OpenKE-PyTorch", "benchmarks", "WN18RR"),
        os.path.join("OpenKE-PyTorch", "benchmarks", "WN18RR"),
    )
    marker = "entity2id.txt"
    for root in search_roots:
        for rp in rel_paths:
            p = os.path.join(root, rp)
            if os.path.isfile(os.path.join(p, marker)):
                return p
        if os.path.isfile(os.path.join(root, marker)):
            return root
    return None


def _copytree(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("Copied:", src, "->", dst)


if not ON_KAGGLE and os.path.isfile(
    os.path.join(WORK, "scripts", "gold_openke", "openke_to_gold.py")
):
    OPENKE_ROOT = WORK

search_roots = []
if ON_KAGGLE:
    if INPUT_BUNDLE and os.path.isdir(INPUT_BUNDLE):
        search_roots.append(os.path.abspath(INPUT_BUNDLE))
    for d in _kaggle_dataset_dirs():
        if d not in search_roots:
            search_roots.append(d)
    print("Kaggle search_roots:", search_roots)

src_openke = _find_openke_repo_root(search_roots)
src_gold = _find_gold_root(search_roots)
src_wn = _find_wn18rr_data_dir(search_roots)

if src_openke and not os.path.isfile(os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "openke_to_gold.py")):
    _copytree(src_openke, OPENKE_ROOT)
if src_gold and not os.path.isfile(os.path.join(GOLD_ROOT, "gold.py")):
    _copytree(src_gold, GOLD_ROOT)
if src_wn:
    if not os.path.isfile(os.path.join(OPENKE_DATA, "entity2id.txt")):
        _copytree(src_wn, OPENKE_DATA)

if GIT_OPENKE and not os.path.isfile(os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "openke_to_gold.py")):
    if os.path.isdir(OPENKE_ROOT):
        shutil.rmtree(OPENKE_ROOT)
    run(f"git clone --depth 1 {GIT_OPENKE} \"{OPENKE_ROOT}\"")
if GIT_GOLD and not os.path.isfile(os.path.join(GOLD_ROOT, "gold.py")):
    if os.path.isdir(GOLD_ROOT):
        shutil.rmtree(GOLD_ROOT)
    run(f"git clone --depth 1 {GIT_GOLD} \"{GOLD_ROOT}\"")

bench = os.path.join(OPENKE_ROOT, "benchmarks", "WN18RR")
if not os.path.isfile(os.path.join(OPENKE_DATA, "entity2id.txt")) and os.path.isdir(bench):
    _copytree(bench, OPENKE_DATA)

if ON_KAGGLE and not os.path.isfile(os.path.join(OPENKE_DATA, "entity2id.txt")):
    thunlp_path = os.path.join(WORK, "_upstream_openke_thunlp")

    def _has_entity2id_somewhere(base_dir: str) -> bool:
        for _, _, files in os.walk(base_dir):
            if "entity2id.txt" in files:
                return True
        return False

    need_clone = not (os.path.isdir(thunlp_path) and _has_entity2id_somewhere(thunlp_path))
    if need_clone:
        print(
            "Không thấy WN18RR trong input — clone thunlp/OpenKE (OpenKE-PyTorch) để lấy benchmarks…"
        )
        if os.path.isdir(thunlp_path):
            shutil.rmtree(thunlp_path)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "OpenKE-PyTorch",
                "https://github.com/thunlp/OpenKE.git",
                thunlp_path,
            ]
        )

    wn_dir = os.path.join(thunlp_path, "benchmarks", "WN18RR")
    if not os.path.isfile(os.path.join(wn_dir, "entity2id.txt")):
        found = None
        for dp, _, files in os.walk(thunlp_path):
            if os.path.basename(dp) == "WN18RR" and "entity2id.txt" in files:
                found = dp
                break
        wn_dir = found

    if not wn_dir:
        raise RuntimeError("Không tìm thấy benchmark WN18RR trong clone OpenKE upstream.")

    print("Using upstream benchmark dir:", wn_dir)
    _copytree(wn_dir, OPENKE_DATA)

    for rf in ["entity2id.txt", "relation2id.txt", "train2id.txt"]:
        if not os.path.isfile(os.path.join(OPENKE_DATA, rf)):
            print("WARN: missing", rf, "in", OPENKE_DATA)

print("=== CHECK REPO STRUCTURE ===")
print(
    "OpenKE scripts/gold_openke/openke_to_gold.py:",
    os.path.isfile(os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "openke_to_gold.py")),
)
print(
    "OpenKE openke/data/PyTorchTrainDataLoader.py:",
    os.path.isfile(os.path.join(OPENKE_ROOT, "openke", "data", "PyTorchTrainDataLoader.py")),
)
print(
    "OpenKE scripts/gold_openke/gold_scores_to_openke_train.py:",
    os.path.isfile(
        os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "gold_scores_to_openke_train.py")
    ),
)
print("GOLD-main gold.py:", os.path.isfile(os.path.join(GOLD_ROOT, "gold.py")))
print("GOLD-main data_loader.py:", os.path.isfile(os.path.join(GOLD_ROOT, "data_loader.py")))

print("OPENKE_DATA exists:", os.path.isdir(OPENKE_DATA))
if os.path.isdir(OPENKE_DATA):
    try:
        print("OPENKE_DATA files:", sorted(os.listdir(OPENKE_DATA))[:20])
    except Exception as e:
        print("WARN listdir OPENKE_DATA:", e)

missing = []
if not os.path.isfile(os.path.join(OPENKE_DATA, "entity2id.txt")):
    missing.append(
        "WN18RR (entity2id.txt …): thêm Dataset benchmark hoặc repo OpenKE có benchmarks/WN18RR"
    )
if not os.path.isfile(os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "openke_to_gold.py")):
    missing.append(
        "OpenKE-OpenKE-PyTorch (có scripts/gold_openke/): upload dataset hoặc GIT_OPENKE="
        "'https://github.com/<you>/OpenKE-OpenKE-PyTorch.git'"
    )
if not os.path.isfile(os.path.join(GOLD_ROOT, "gold.py")):
    missing.append(
        "GOLD-main: upload dataset hoặc GIT_GOLD='https://github.com/<you>/GOLD-main.git'"
    )

if missing:
    kaggle_hint = ""
    if ON_KAGGLE:
        inp = "/kaggle/input"
        exists = os.path.isdir(inp)
        inside = os.listdir(inp) if exists else []
        kaggle_hint = (
            "\n\n[Chẩn đoán Kaggle] /kaggle/input tồn tại: %s | Nội dung: %s\n"
            "Nếu list rỗng: tab notebook → Input → Add input → Datasets → Save → Run lại cell."
            % (exists, inside)
        )
    raise RuntimeError(
        "Thiếu OpenKE/GOLD hoặc benchmark WN18RR.\n"
        "Bật Internet và đặt GIT_OPENKE / GIT_GOLD ở cell cấu hình nếu chưa gắn Input.\n\n"
        + "\n".join(" - " + m for m in missing)
        + "\n\nĐã quét: "
        + repr(search_roots if ON_KAGGLE else [WORK])
        + kaggle_hint
    )

print("OK: code + data found")
print("  OPENKE_ROOT:", OPENKE_ROOT)
print("  GOLD_ROOT:", GOLD_ROOT)
print("  OPENKE_DATA:", OPENKE_DATA)
```

## Cell 3 — code

```python
run([
    sys.executable,
    os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "openke_to_gold.py"),
    "--openke_dir", OPENKE_DATA,
    "--out_dir", GOLD_LAYOUT,
    "--dataset_tag", DATASET_TAG,
    "--noise_ratio", "0.05",
    "--rule_top_k", str(RULE_TOP_K),
    "--seed", "42",
])
```

## Cell 4 — code

```python
# Cài dependency cho GOLD (không patch runtime)
run("apt-get -qq update && apt-get -qq install -y build-essential")
run(f'"{sys.executable}" -m pip install -q -U pip setuptools wheel')

import sys as _sys
print("Python version:", _sys.version)

try:
    import torch
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| CC:", torch.cuda.get_device_capability(0))
except Exception:
    if ON_KAGGLE:
        run(
            '"' + _sys.executable + '" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121'
        )
    else:
        run(
            '"' + _sys.executable + '" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu'
        )

run(f'"{_sys.executable}" -m pip install -q sentence-transformers tqdm')
print("Installed: sentence-transformers, tqdm")
```

## Cell 5 — code

```python
gold_tsv = os.path.join(GOLD_ROOT, f"{DATASET_TAG}_{MODEL_NAME}.tsv")
_gold_cmd = [
    sys.executable,
    os.path.join(GOLD_ROOT, "gold.py"),
    "--dataset", DATASET_TAG,
    "--dataset_path", GOLD_LAYOUT,
    "--model_name", MODEL_NAME,
    "--epoch", str(EPOCH),
    "--batch_size", str(GOLD_BATCH_SIZE),
    "--topk", str(RULE_TOP_K),
    "--ptlm_model", GOLD_PTLM_MODEL,
    "--lr", str(GOLD_LR),
    "--local_lambda", str(GOLD_LOCAL_LAMBDA),
    "--global_lambda", str(GOLD_GLOBAL_LAMBDA),
    "--neg_cnt", str(GOLD_NEG_CNT),
    "--seed", str(GOLD_SEED),
    "--output_tsv",
    "--device", "auto",
]
print("$", " ".join(_gold_cmd))
_rg = subprocess.run(_gold_cmd, cwd=GOLD_ROOT, capture_output=True, text=True)
if _rg.stdout:
    print(_rg.stdout, end="" if _rg.stdout.endswith("\n") else "\n")
if _rg.stderr:
    print(_rg.stderr, end="" if _rg.stderr.endswith("\n") else "\n", file=sys.stderr)
if _rg.returncode != 0:
    raise RuntimeError("gold.py failed")

_combined = (_rg.stdout or "") + (_rg.stderr or "")
GOLD_METRICS = parse_gold_stdout(_combined)
print("GOLD_METRICS (for results file):", GOLD_METRICS)

assert os.path.isfile(gold_tsv), f"Không thấy TSV: {gold_tsv}"
print("TSV:", gold_tsv)
```

## Cell 6 — code (lọc train + thống kê triple)

In ra log chi tiết (từ `gold_scores_to_openke_train.py`) và ghi thêm **`OPENKE_DATA/gold_train_filter_stats.txt`** để lưu cùng artifact (Cell 9 zip).

```python
clean_train = os.path.join(OPENKE_DATA, "train2id_gold_clean.txt")
GOLD_TRAIN_FILTER_STATS = os.path.join(OPENKE_DATA, "gold_train_filter_stats.txt")

_filter_cmd = [
    sys.executable,
    os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "gold_scores_to_openke_train.py"),
    "--gold_tsv", gold_tsv,
    "--openke_dir", OPENKE_DATA,
    "--out_train2id", clean_train,
    "--drop_top_fraction", "0.05",
]
print("$", " ".join(_filter_cmd))
_rf = subprocess.run(_filter_cmd, capture_output=True, text=True)
if _rf.stdout:
    print(_rf.stdout, end="" if _rf.stdout.endswith("\n") else "\n")
if _rf.stderr:
    print(_rf.stderr, end="" if _rf.stderr.endswith("\n") else "\n", file=sys.stderr)
if _rf.returncode != 0:
    raise RuntimeError("gold_scores_to_openke_train.py failed")

with open(GOLD_TRAIN_FILTER_STATS, "w", encoding="utf-8") as _sf:
    _sf.write((_rf.stdout or "").strip() + "\n")
print("Đã ghi thống kê lọc train:", GOLD_TRAIN_FILTER_STATS)


def _read_train2id_n(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as _f:
        return int(_f.readline().strip())


_n_orig = _read_train2id_n(os.path.join(OPENKE_DATA, "train2id.txt"))
_n_clean = _read_train2id_n(clean_train)
if _n_orig is not None and _n_clean is not None:
    print(
        "Tóm tắt nhanh (dòng 1 train2id): train gốc = %d | train sau GOLD = %d | không vào file sạch = %d"
        % (_n_orig, _n_clean, _n_orig - _n_clean)
    )

assert os.path.isfile(clean_train), "Không thấy file train đã lọc"
print("Clean train:", clean_train)
```

## Cell 7 — code

```python
import platform
make_dir = os.path.join(OPENKE_ROOT, "openke")
if ON_KAGGLE or platform.system() == "Linux":
    run("apt-get -qq update && apt-get -qq install -y build-essential")
    run("bash make.sh", cwd=make_dir)
else:
    print("Bỏ qua make.sh — trên Windows hãy biên dịch Base theo README OpenKE hoặc chạy cell này trên Kaggle/Linux.")
base_so = os.path.join(OPENKE_ROOT, "openke", "release", "Base.so")
print("Base.so exists:", os.path.isfile(base_so))
```

## Cell 8 — code

```python
sys.path.insert(0, OPENKE_ROOT)
os.chdir(OPENKE_ROOT)

import torch
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TestDataLoader
from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader

DATASET_PATH = os.path.abspath(OPENKE_DATA).replace("\\", "/") + "/"

USE_GPU = torch.cuda.is_available()
print("USE_GPU:", USE_GPU)

train_dataloader = PyTorchTrainDataLoader(
    in_path=DATASET_PATH,
    tri_file=clean_train,
    nbatches=TRANSE_NBATCHES,
    threads=TRANSE_THREADS,
    sampling_mode="normal",
    bern_flag=TRANSE_BERN_FLAG,
    filter_flag=TRANSE_FILTER_FLAG,
    neg_ent=TRANSE_NEG_ENT,
    neg_rel=0,
)
test_dataloader = TestDataLoader(DATASET_PATH, "link")

transe = TransE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=TRANSE_DIM,
    p_norm=1,
    norm_flag=True,
)
model = NegativeSampling(
    model=transe,
    loss=MarginLoss(margin=TRANSE_MARGIN),
    batch_size=train_dataloader.get_batch_size(),
)

ckpt_dir = os.path.join(OPENKE_ROOT, "checkpoint_gold_clean_wn18rr")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "transe_gold_clean.ckpt")

_resume_path = TRANSE_RESUME_CKPT
_resume_latest = bool(TRANSE_RESUME_LATEST) and not _resume_path

trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=TRAIN_TIMES,
    alpha=TRANSE_ALPHA,
    use_gpu=USE_GPU,
    save_steps=TRANSE_SAVE_EVERY if TRANSE_SAVE_EVERY else None,
    checkpoint_dir=ckpt_dir,
    checkpoint_prefix=TRANSE_CHECKPOINT_PREFIX,
    save_model=transe,
    resume_from_checkpoint=_resume_path,
    resume_from_latest=_resume_latest,
)
trainer.run()
transe.save_checkpoint(ckpt_path)
print("Saved:", ckpt_path)

transe.load_checkpoint(ckpt_path)
tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=USE_GPU)
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=TRANSE_TYPE_CONSTRAIN)

import datetime

try:
    _gm = GOLD_METRICS
except NameError:
    _gm = {}


def _fmt_metric(v):
    return ("%f" % v) if v is not None else "NA"


results_txt = os.path.join(ckpt_dir, "transe_training_results.txt")
_lines = [
    "TransE + link prediction (WN18RR + GOLD filtered train)",
    "time_utc: %s" % (datetime.datetime.utcnow().isoformat() + "Z",),
    "dataset_tag: %s  model_name: %s" % (DATASET_TAG, MODEL_NAME),
    "train_times: %s  dim: %s  nbatches: %s  neg_ent: %s" % (
        TRAIN_TIMES, TRANSE_DIM, TRANSE_NBATCHES, TRANSE_NEG_ENT
    ),
    "DATASET_PATH: %s" % (DATASET_PATH,),
    "clean_train: %s" % (clean_train,),
    "checkpoint: %s" % (ckpt_path,),
    "",
    "=== GOLD (gold.py valid/test Acc & AUC) ===",
    "gold_valid_Acc\t%s" % _fmt_metric(_gm.get("gold_valid_acc")),
    "gold_valid_AUC\t%s" % _fmt_metric(_gm.get("gold_valid_auc")),
    "gold_test_Acc\t%s" % _fmt_metric(_gm.get("gold_test_acc")),
    "gold_test_AUC\t%s" % _fmt_metric(_gm.get("gold_test_auc")),
    "",
    "=== OpenKE link prediction (TransE) ===",
    "MRR\t%f" % (mrr,),
    "MR\t%f" % (mr,),
    "Hits@10\t%f" % (hit10,),
    "Hits@3\t%f" % (hit3,),
    "Hits@1\t%f" % (hit1,),
    "",
]
with open(results_txt, "w", encoding="utf-8") as _f:
    _f.write("\n".join(_lines))
print("Wrote results:", results_txt)
```

## Cell 9 — code

```python
if ON_KAGGLE:
    out_zip = "/kaggle/working/gold_openke_wn18rr_artifact.zip"
    _zip_parts = [clean_train, ckpt_path, gold_tsv, results_txt]
    _stats_zip = os.path.join(OPENKE_DATA, "gold_train_filter_stats.txt")
    if os.path.isfile(_stats_zip):
        _zip_parts.append(_stats_zip)
    run('zip -r "%s" %s' % (out_zip, " ".join('"%s"' % p for p in _zip_parts)))
    print("Created", out_zip)
else:
    print("Skip zip (not on Kaggle)")
```
