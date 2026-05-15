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
elif os.path.isfile(os.path.join(_cwd, "scripts", "gold_openke", "conceptnet_to_openke.py")):
    WORK = os.path.abspath(os.path.join(_cwd, "..", ".."))
else:
    WORK = os.path.abspath(_cwd)

INPUT_BUNDLE = None
GIT_OPENKE = None
GIT_GOLD = None

OPENKE_ROOT = os.path.join(WORK, "OpenKE-OpenKE-PyTorch")
GOLD_ROOT = os.path.join(WORK, "GOLD-main")

CONCEPTNET_DIR = os.path.join(GOLD_ROOT, "dataset", "conceptnet")
OPENKE_DATA = os.path.join(WORK, "data", "ConceptNet_openke")

DATASET = "C-05"
MODEL_NAME = "cn_openke1"
RULE_TOP_K = 100
PTLM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# GOLD (gold.py): đặt 1 để chạy nhanh; 10 cho huấn luyện đầy đủ.
EPOCH = 10
BATCH_SIZE = 128
DROP_TOP = 0.05
TRAIN_TIMES = 300
# TransE — ConceptNet lớn: batch nhỏ (nbatches cao), dim thấp, neg ít; script subprocess tắt pin_memory mặc định.
# Vẫn OOM → giảm TRAIN_TIMES / TRANSE_DIM hoặc thêm --use_cpu trong cell TransE.
TRANSE_DIM = 64
TRANSE_NBATCHES = 480
TRANSE_NEG_ENT = 3
TRANSE_MARGIN = 4.0
TRANSE_ALPHA = 0.3
# Bật True nếu vẫn OOM GPU (TransE trên CPU — rất chậm).
TRANSE_USE_CPU = False

gold_tsv = os.path.join(GOLD_ROOT, "conceptnet_%s.tsv" % MODEL_NAME)
clean_train = os.path.join(OPENKE_DATA, "train2id_gold_clean.txt")
train_txt_for_restrict = os.path.join(CONCEPTNET_DIR, "train.txt")

os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.dirname(OPENKE_DATA), exist_ok=True)

print("WORK:", WORK)
print("GOLD_ROOT:", GOLD_ROOT)
print("OPENKE_DATA:", OPENKE_DATA)
print("gold_tsv:", gold_tsv)
```

## Cell 2 — code

```python
import os
import shutil
import subprocess
import sys

def run(cmd, cwd=None):
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    if isinstance(cmd, str):
        r = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    else:
        r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            "Command failed (exit %s). Xem log phía trên. Gợi ý: -9 hoặc 137 → thường OOM "
            "(thử TRANSE_USE_CPU=True, giảm TRAIN_TIMES/TRANSE_DIM); -11 → SIGSEGV (OpenKE/native; "
            "với ConceptNet cần run_transe không dùng type_constrain khi thiếu type_constrain.txt)."
            % (r.returncode,)
        )


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
    subnames = ("OpenKE-OpenKE-PyTorch", "OpenKE-PyTorch")
    for root in search_roots:
        for sub in subnames:
            p = os.path.join(root, sub)
            if os.path.isfile(os.path.join(p, "scripts", "gold_openke", "conceptnet_to_openke.py")):
                return p
        if os.path.isfile(os.path.join(root, "scripts", "gold_openke", "conceptnet_to_openke.py")):
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


def _copytree(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("Copied:", src, "->", dst)


if not ON_KAGGLE and os.path.isfile(
    os.path.join(WORK, "scripts", "gold_openke", "conceptnet_to_openke.py")
):
    OPENKE_ROOT = WORK

search_roots = [WORK]
if ON_KAGGLE:
    if INPUT_BUNDLE and os.path.isdir(INPUT_BUNDLE):
        search_roots.append(os.path.abspath(INPUT_BUNDLE))
    for d in _kaggle_dataset_dirs():
        if d not in search_roots:
            search_roots.append(d)
    print("Kaggle search_roots:", search_roots)

src_openke = _find_openke_repo_root(search_roots)
src_gold = _find_gold_root(search_roots)

if src_openke and not os.path.isfile(os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "conceptnet_to_openke.py")):
    _copytree(src_openke, OPENKE_ROOT)
if src_gold and not os.path.isfile(os.path.join(GOLD_ROOT, "gold.py")):
    _copytree(src_gold, GOLD_ROOT)

if GIT_OPENKE and not os.path.isfile(os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "conceptnet_to_openke.py")):
    if os.path.isdir(OPENKE_ROOT):
        shutil.rmtree(OPENKE_ROOT)
    run(["git", "clone", "--depth", "1", GIT_OPENKE, OPENKE_ROOT])
if GIT_GOLD and not os.path.isfile(os.path.join(GOLD_ROOT, "gold.py")):
    if os.path.isdir(GOLD_ROOT):
        shutil.rmtree(GOLD_ROOT)
    run(["git", "clone", "--depth", "1", GIT_GOLD, GOLD_ROOT])

missing = []
if not os.path.isfile(os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "conceptnet_to_openke.py")):
    missing.append("OpenKE repo (conceptnet_to_openke.py)")
if not os.path.isfile(os.path.join(GOLD_ROOT, "gold.py")):
    missing.append("GOLD-main gold.py")
if not os.path.isfile(os.path.join(CONCEPTNET_DIR, "train.txt")):
    missing.append("GOLD dataset/conceptnet/train.txt")

if missing:
    raise RuntimeError("Thieu: " + ", ".join(missing) + "\nAdd Kaggle Input hoac GIT_OPENKE / GIT_GOLD.")

print("OK OPENKE_ROOT:", OPENKE_ROOT)
print("OK GOLD_ROOT:", GOLD_ROOT)
```

## Cell 3 — code

```python
run([
    sys.executable,
    os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "conceptnet_to_openke.py"),
    "--conceptnet_dir", CONCEPTNET_DIR,
    "--out_dir", OPENKE_DATA,
])
```

## Cell 4 — code

```python
run("apt-get -qq update && apt-get -qq install -y build-essential")
run('"%s" -m pip install -q -U pip setuptools wheel' % sys.executable)

try:
    import torch
    print("torch:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| CC:", torch.cuda.get_device_capability(0))
except Exception:
    if ON_KAGGLE:
        run('"%s" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121' % sys.executable)
    else:
        run('"%s" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu' % sys.executable)

# Trên Kaggle: KHÔNG pip -r GOLD requirements.txt (numpy/torch pin cũ → lỗi build wheel).
run('"%s" -m pip install -q numpy scikit-learn tqdm' % sys.executable)
run('"%s" -m pip install -q sentence-transformers' % sys.executable)
print("pip OK (numpy/sklearn/tqdm/sentence-transformers; torch bản Kaggle GPU hoặc cu121 nếu phải cài lại)")
```

## Cell 5 — code

```python
import re

_gold_cmd = [
    sys.executable,
    os.path.join(GOLD_ROOT, "gold.py"),
    "--dataset", DATASET,
    "--model_name", MODEL_NAME,
    "--epoch", str(EPOCH),
    "--batch_size", str(BATCH_SIZE),
    "--topk", str(RULE_TOP_K),
    "--ptlm_model", PTLM_MODEL,
    "--lr", "0.001",
    "--local_lambda", "0.1",
    "--global_lambda", "0.01",
    "--neg_cnt", "1",
    "--seed", "5",
    "--output_tsv",
    "--device", "auto",
]

_env = dict(os.environ)
_env.setdefault("PYTHONUNBUFFERED", "1")
_env.setdefault("PYTHONFAULTHANDLER", "1")
print("$", " ".join(_gold_cmd))
_r = subprocess.run(_gold_cmd, cwd=GOLD_ROOT, env=_env, capture_output=True, text=True)
if _r.stdout:
    print(_r.stdout, end="")
if _r.stderr:
    print(_r.stderr, end="")
if _r.returncode != 0:
    raise RuntimeError("gold.py failed (exit %s)" % _r.returncode)

_gold_text = ((_r.stdout or "") + "\n" + (_r.stderr or ""))
_valid = re.findall(r"\[ACC REPORT\][^\n]*Acc = ([0-9.]+),\s*AUC = ([0-9.]+)", _gold_text)
_test = re.findall(r"\[TEST RESULT\][^\n]*Acc = ([0-9.]+),\s*AUC = ([0-9.]+)", _gold_text)

GOLD_METRICS = {}
if _valid:
    GOLD_METRICS["gold_valid_acc"] = float(_valid[-1][0])
    GOLD_METRICS["gold_valid_auc"] = float(_valid[-1][1])
if _test:
    GOLD_METRICS["gold_test_acc"] = float(_test[-1][0])
    GOLD_METRICS["gold_test_auc"] = float(_test[-1][1])

print("GOLD_METRICS:", GOLD_METRICS)
assert os.path.isfile(gold_tsv), "Khong thay TSV: %s" % gold_tsv
print("TSV:", gold_tsv)
```

## Cell 6 — code

```python
GOLD_TRAIN_FILTER_STATS = os.path.join(OPENKE_DATA, "gold_train_filter_stats.txt")

_filter_cmd = [
    sys.executable,
    os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "gold_scores_to_openke_train.py"),
    "--gold_tsv", gold_tsv,
    "--openke_dir", OPENKE_DATA,
    "--out_train2id", clean_train,
    "--drop_top_fraction", str(DROP_TOP),
    "--restrict_train_triples_txt", train_txt_for_restrict,
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
print("Da ghi thong ke loc train:", GOLD_TRAIN_FILTER_STATS)


def _read_train2id_n(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as _f:
        return int(_f.readline().strip())


_n_orig = _read_train2id_n(os.path.join(OPENKE_DATA, "train2id.txt"))
_n_clean = _read_train2id_n(clean_train)
if _n_orig is not None and _n_clean is not None:
    print(
        "Tom tat nhanh (dong 1 train2id): train goc = %d | train sau GOLD = %d | khong vao file sach = %d"
        % (_n_orig, _n_clean, _n_orig - _n_clean)
    )

assert os.path.isfile(clean_train), "Khong thay file train da loc"
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
    print("Bo qua make.sh tren Windows — dung Kaggle/Linux hoac build Base theo README OpenKE.")
print("Base.so:", os.path.isfile(os.path.join(make_dir, "release", "Base.so")))
```

## Cell 8 — code

```python
import gc
import datetime

gc.collect()
try:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass

ckpt_dir = os.path.join(OPENKE_ROOT, "checkpoint_conceptnet_gold")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "transe_cn_gold.ckpt")
transe_results_txt = os.path.join(ckpt_dir, "transe_training_results.txt")

_transe_cmd = [
    sys.executable,
    "-u",  # unbuffered: traceback / print luôn lên log Kaggle
    os.path.join(OPENKE_ROOT, "scripts", "gold_openke", "run_transe_conceptnet_openke.py"),
    "--openke_root",
    OPENKE_ROOT,
    "--data_path",
    OPENKE_DATA,
    "--tri_file",
    clean_train,
    "--train_times",
    str(TRAIN_TIMES),
    "--dim",
    str(TRANSE_DIM),
    "--nbatches",
    str(TRANSE_NBATCHES),
    "--neg_ent",
    str(TRANSE_NEG_ENT),
    "--margin",
    str(TRANSE_MARGIN),
    "--alpha",
    str(TRANSE_ALPHA),
    "--ckpt_out",
    ckpt_path,
    "--metrics_out",
    transe_results_txt,
]
if TRANSE_USE_CPU:
    _transe_cmd.append("--use_cpu")

run(_transe_cmd)
assert os.path.isfile(ckpt_path), "Thieu checkpoint sau TransE subprocess"
assert os.path.isfile(transe_results_txt), "Thieu file metrics TransE"

try:
    _gm = GOLD_METRICS
except NameError:
    _gm = {}

def _fmt_metric(v):
    return ("%f" % v) if v is not None else "NA"

with open(transe_results_txt, "r", encoding="utf-8") as _f:
    _raw_openke = _f.read().strip()

_lines = [
    "TransE + GOLD summary (ConceptNet)",
    "time_utc: %s" % (datetime.datetime.utcnow().isoformat() + "Z",),
    "dataset: %s  model_name: %s" % (DATASET, MODEL_NAME),
    "train_times: %s  dim: %s  nbatches: %s  neg_ent: %s  margin: %s  alpha: %s"
    % (TRAIN_TIMES, TRANSE_DIM, TRANSE_NBATCHES, TRANSE_NEG_ENT, TRANSE_MARGIN, TRANSE_ALPHA),
    "data_path: %s" % OPENKE_DATA,
    "clean_train: %s" % clean_train,
    "checkpoint: %s" % ckpt_path,
    "",
    "=== GOLD (gold.py valid/test Acc & AUC) ===",
    "gold_valid_Acc\t%s" % _fmt_metric(_gm.get("gold_valid_acc")),
    "gold_valid_AUC\t%s" % _fmt_metric(_gm.get("gold_valid_auc")),
    "gold_test_Acc\t%s" % _fmt_metric(_gm.get("gold_test_acc")),
    "gold_test_AUC\t%s" % _fmt_metric(_gm.get("gold_test_auc")),
    "",
    "=== OpenKE link prediction (raw metrics) ===",
    _raw_openke,
    "",
]
with open(transe_results_txt, "w", encoding="utf-8") as _f:
    _f.write("\n".join(_lines))

print("OK:", ckpt_path)
print("Wrote results:", transe_results_txt)
```

## Cell 9 — code

```python
if ON_KAGGLE:
    out_zip = "/kaggle/working/gold_cn_openke_artifact.zip"
    _z = [clean_train, ckpt_path, gold_tsv, transe_results_txt]
    _stats_zip = os.path.join(OPENKE_DATA, "gold_train_filter_stats.txt")
    if os.path.isfile(_stats_zip):
        _z.append(_stats_zip)
    run('zip -r "%s" %s' % (out_zip, " ".join('"%s"' % p for p in _z)))
    print("Created", out_zip)
else:
    print("Skip zip")
```
