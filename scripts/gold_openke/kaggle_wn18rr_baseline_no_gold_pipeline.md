# Kaggle — WN18RR + TransE baseline (không GOLD)

Notebook **so sánh** với [`kaggle_wn18rr_gold_openke_pipeline.md`](kaggle_wn18rr_gold_openke_pipeline.md): cùng benchmark **WN18RR**, cùng siêu tham số TransE mặc định bên dưới, nhưng train trên **`train2id.txt` chuẩn OpenKE** — không `openke_to_gold.py`, không `gold.py`, không `gold_scores_to_openke_train.py`.

**Accelerator:** GPU T4 (Kaggle). Không cần `sentence-transformers` / GOLD-main.

**Input:** dataset có repo OpenKE (hoặc chỉ bật **Internet** + `GIT_OPENKE`) và dữ liệu WN18RR (`entity2id.txt` …) giống pipeline GOLD.

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
elif os.path.isfile(os.path.join(_cwd, "openke", "data", "PyTorchTrainDataLoader.py")):
    WORK = os.path.abspath(_cwd)
elif os.path.isfile(os.path.join(_cwd, "scripts", "gold_openke", "openke_to_gold.py")):
    WORK = os.path.abspath(os.path.join(_cwd, "..", ".."))
else:
    WORK = os.path.abspath(_cwd)

# ===== Hyper-parameters (chỉnh tập trung ở đây; nên giữ giống run GOLD để so sánh) =====
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

# File train trong OPENKE_DATA (OpenKE mặc định)
TRANSE_TRAIN_FILE = "train2id.txt"

TRANSE_SAVE_EVERY = 100
TRANSE_CHECKPOINT_PREFIX = "transe_baseline_wn18rr"
TRANSE_RESUME_LATEST = False
TRANSE_RESUME_CKPT = None

INPUT_BUNDLE = None
GIT_OPENKE = "https://github.com/Congvinh2004/ML_KGE.git"

OPENKE_ROOT = os.path.join(WORK, "OpenKE-OpenKE-PyTorch")
OPENKE_DATA = os.path.join(WORK, "data", "WN18RR")

DATASET_TAG = "WN18RR_openke"
RUN_TAG = "kaggle_wn18rr_baseline_no_gold"

os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.dirname(OPENKE_DATA), exist_ok=True)

print("WORK:", WORK)
print("ON_KAGGLE:", ON_KAGGLE)
print("RUN_TAG:", RUN_TAG)
print("TRANSE_TRAIN_FILE:", TRANSE_TRAIN_FILE)
```

---

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
    """Thư mục repo OpenKE (có openke/data/PyTorchTrainDataLoader.py)."""
    subnames = ("OpenKE-OpenKE-PyTorch", "OpenKE-PyTorch")
    marker = os.path.join("openke", "data", "PyTorchTrainDataLoader.py")
    for root in search_roots:
        for sub in subnames:
            p = os.path.join(root, sub)
            if os.path.isfile(os.path.join(p, marker)):
                return p
        if os.path.isfile(os.path.join(root, marker)):
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
    os.path.join(WORK, "openke", "data", "PyTorchTrainDataLoader.py")
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
src_wn = _find_wn18rr_data_dir(search_roots)

_openke_marker = os.path.join(OPENKE_ROOT, "openke", "data", "PyTorchTrainDataLoader.py")
if src_openke and not os.path.isfile(_openke_marker):
    _copytree(src_openke, OPENKE_ROOT)
if src_wn and not os.path.isfile(os.path.join(OPENKE_DATA, "entity2id.txt")):
    _copytree(src_wn, OPENKE_DATA)

if GIT_OPENKE and not os.path.isfile(_openke_marker):
    if os.path.isdir(OPENKE_ROOT):
        shutil.rmtree(OPENKE_ROOT)
    run(f"git clone --depth 1 {GIT_OPENKE} \"{OPENKE_ROOT}\"")

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

TRAIN_PATH = os.path.join(OPENKE_DATA, TRANSE_TRAIN_FILE)

print("=== CHECK (baseline, no GOLD) ===")
print(
    "OpenKE PyTorchTrainDataLoader.py:",
    os.path.isfile(_openke_marker),
)
print("OPENKE_DATA exists:", os.path.isdir(OPENKE_DATA))
if os.path.isdir(OPENKE_DATA):
    try:
        print("OPENKE_DATA files:", sorted(os.listdir(OPENKE_DATA))[:25])
    except Exception as e:
        print("WARN listdir OPENKE_DATA:", e)

missing = []
if not os.path.isfile(os.path.join(OPENKE_DATA, "entity2id.txt")):
    missing.append(
        "WN18RR (entity2id.txt …): thêm Dataset benchmark hoặc repo OpenKE có benchmarks/WN18RR"
    )
if not os.path.isfile(TRAIN_PATH):
    missing.append("Train file %s trong OPENKE_DATA" % TRANSE_TRAIN_FILE)
if not os.path.isfile(_openke_marker):
    missing.append(
        "OpenKE repo (openke/data/PyTorchTrainDataLoader.py): upload dataset hoặc GIT_OPENKE="
        "'https://github.com/<you>/OpenKE-OpenKE-PyTorch.git'"
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
        "Thiếu OpenKE hoặc benchmark WN18RR.\n"
        "Bật Internet và đặt GIT_OPENKE ở cell cấu hình nếu chưa gắn Input.\n\n"
        + "\n".join(" - " + m for m in missing)
        + "\n\nĐã quét: "
        + repr(search_roots if ON_KAGGLE else [WORK])
        + kaggle_hint
    )

print("OK: code + data found")
print("  OPENKE_ROOT:", OPENKE_ROOT)
print("  OPENKE_DATA:", OPENKE_DATA)
print("  TRAIN_PATH:", TRAIN_PATH)
```

---

## Cell 3 — code

```python
# Kiểm tra torch (Kaggle thường đã có CUDA). Máy local không GPU: cài CPU wheel.
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
```

---

## Cell 4 — code

```python
import platform

make_dir = os.path.join(OPENKE_ROOT, "openke")
if ON_KAGGLE or platform.system() == "Linux":
    run("apt-get -qq update && apt-get -qq install -y build-essential")
    run("bash make.sh", cwd=make_dir)
else:
    print(
        "Bỏ qua make.sh — trên Windows hãy biên dịch Base theo README OpenKE hoặc chạy cell này trên Kaggle/Linux."
    )
base_so = os.path.join(OPENKE_ROOT, "openke", "release", "Base.so")
print("Base.so exists:", os.path.isfile(base_so))
```

---

## Cell 5 — code

```python
sys.path.insert(0, OPENKE_ROOT)
os.chdir(OPENKE_ROOT)

import datetime
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
    tri_file=TRAIN_PATH,
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

ckpt_dir = os.path.join(OPENKE_ROOT, "checkpoint_baseline_wn18rr")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "transe_baseline.ckpt")

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

results_txt = os.path.join(ckpt_dir, "transe_training_results.txt")
_train_display = TRAIN_PATH.replace("\\", "/")
_ckpt_display = ckpt_path.replace("\\", "/")
_lines = [
    "TransE + link prediction (WN18RR baseline, no GOLD — standard train2id.txt)",
    "time_utc: %s" % (datetime.datetime.utcnow().isoformat() + "Z",),
    "run_tag: %s  dataset_tag: %s" % (RUN_TAG, DATASET_TAG),
    "train_times: %s  dim: %s  nbatches: %s  neg_ent: %s"
    % (TRAIN_TIMES, TRANSE_DIM, TRANSE_NBATCHES, TRANSE_NEG_ENT),
    "DATASET_PATH: %s" % (DATASET_PATH,),
    "train_file: %s" % (_train_display,),
    "checkpoint: %s" % (_ckpt_display,),
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
print("MRR", mrr, "MR", mr, "Hits@10", hit10, "Hits@3", hit3, "Hits@1", hit1)
```

---

## Cell 6 — code

```python
if ON_KAGGLE:
    out_zip = "/kaggle/working/openke_wn18rr_baseline_artifact.zip"
    run('zip -r "%s" "%s" "%s"' % (out_zip, ckpt_path, results_txt))
    print("Created", out_zip)
else:
    print("Skip zip (not on Kaggle)")
```

---

## Ghi chú so sánh với pipeline GOLD

| Khía cạnh | Pipeline GOLD | Pipeline này (baseline) |
|-----------|---------------|-------------------------|
| Train OpenKE | `train2id_gold_clean.txt` (sau GOLD + filter) | `train2id.txt` gốc |
| GOLD-main | Cần | Không cần |
| `sentence-transformers` | Cần (GOLD) | Không cần |
| Checkpoint / thư mục | `checkpoint_gold_clean_wn18rr/`, `transe_gold_clean.ckpt` | `checkpoint_baseline_wn18rr/`, `transe_baseline.ckpt` |
| Metric `gold.py` (Acc/AUC) | Có | Không |

Đánh giá link prediction vẫn dùng **cùng** `valid2id.txt` / `test2id.txt` trong `OPENKE_DATA` (chuẩn WN18RR OpenKE).
