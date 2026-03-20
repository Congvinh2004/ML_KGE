# Tích hợp GOLD (repo ngoài) với OpenKE

GOLD được thiết kế cho ConceptNet/ATOMIC. Các script này đóng gói dữ liệu OpenKE sang layout GOLD (text triple + file `errors/` + `rules/`), sau đó bạn chạy `gold.py` trong repo GOLD. Cuối cùng, chấm điểm TSV từ GOLD được lọc và xuất lại `train2id*.txt` cho OpenKE.

## 1. Chuẩn bị thư mục OpenKE (chuẩn benchmark)

Trong thư mục dataset cần có tối thiểu:

- `entity2id.txt`
- `relation2id.txt`
- `train2id.txt`

Nếu không có `valid2id.txt` / `test2id.txt`, script sẽ **chia ngẫu nhiên** từ `train2id.txt` (mặc định 80% / 10% / 10%).

## 2. Export sang layout GOLD

Chạy từ repo **OpenKE** (thư mục gốc chứa `scripts/`):

```bash
python scripts/gold_openke/openke_to_gold.py ^
  --openke_dir "D:\path\to\FB15K237" ^
  --out_dir "D:\path\to\GOLD-main\dataset\FB15K237_openke" ^
  --dataset_tag FB15K237_openke ^
  --noise_ratio 0.05 ^
  --rule_top_k 100 ^
  --seed 42
 ```

- `--noise_ratio`: tỷ lệ triple **giả lập nhiễu** (GOLD cần file `errors/*.txt` chứa triple sai để huấn luyện kiểu gốc).
- `--rule_top_k` phải trùng `--topk` khi gọi `gold.py`.

## 3. Chạy GOLD

Trong thư mục repo **GOLD** (ví dụ `GOLD-main`):

```bash
cd D:\CongVinh\Documents\Documents\00_University\KLTN\source\GOLD-main
pip install -r requirements.txt

python gold.py ^
  --dataset FB15K237_openke ^
  --dataset_path dataset/FB15K237_openke ^
  --model_name openke_run1 ^
  --epoch 10 ^
  --batch_size 256 ^
  --topk 100 ^
  --ptlm_model sentence-transformers/all-MiniLM-L6-v2 ^
  --lr 0.001 ^
  --local_lambda 0.1 ^
  --global_lambda 0.01 ^
  --neg_cnt 1 ^
  --seed 5 ^
  --output_tsv ^
  --device auto
```

- Model PTLM nhẹ hơn (ví dụ `all-MiniLM-L6-v2`) dễ chạy hơn mặc định `sentence-t5-xxl`.
- Checkpoint & TSV ghi tại thư mục hiện tại của GOLD. Với patch `--dataset_path` (OpenKE): **`{dataset_tag}_{model_name}.tsv`** — ví dụ `FB15K237_openke_openke_run1.tsv`.

> **Lưu ý:** Dataset benchmark gốc (C-05, A-05, …): tên file vẫn là `{conceptnet|atomic}_{model_name}.tsv` như bản gốc.

## 4. Lọc triple & xuất lại `train2id` cho OpenKE

```bash
python scripts/gold_openke/gold_scores_to_openke_train.py ^
  --gold_tsv "D:\path\to\GOLD-main\FB15K237_openke_openke_run1.tsv" ^
  --openke_dir "D:\path\to\FB15K237" ^
  --out_train2id "D:\path\to\FB15K237\train2id_gold_clean.txt" ^
  --drop_top_fraction 0.05
```

- `--drop_top_fraction 0.05`: bỏ **5%** triple có **điểm (loss) cao nhất** (GOLD coi là khả năng nhiễu cao hơn).

## 5. Huấn luyện OpenKE

Cách 1 — **copy file** (đơn giản): đặt `train2id_gold_clean.txt` đè hoặc đổi tên thành `train2id.txt` trong thư mục benchmark, giữ `in_path` như cũ.

Cách 2 — **chỉ định `tri_file`** (đã hỗ trợ cùng lúc với `in_path` trong `PyTorchTrainDataLoader`):

```python
train_dataloader = PyTorchTrainDataLoader(
    in_path="./benchmarks/FB15K237/",
    tri_file="./benchmarks/FB15K237/train2id_gold_clean.txt",
    nbatches=100,
    threads=0,
    ...
)
```

`entity2id.txt` / `relation2id.txt` / `valid2id.txt` / `test2id.txt` vẫn lấy theo `in_path`; chỉ tập train thay bằng bản đã lọc GOLD.

---

## 6. Chạy toàn pipeline trên Kaggle

**Notebook từng cell (upload lên Kaggle hoặc chạy local):** [`kaggle_gold_openke_pipeline.ipynb`](kaggle_gold_openke_pipeline.ipynb).

Ý tưởng: một notebook GPU, thư mục làm việc `/kaggle/working`; dữ liệu có thể clone hoặc gắn **Kaggle Dataset** (zip chứa `OpenKE-OpenKE-PyTorch` + `GOLD-main` + thư mục `FB15K237` chuẩn OpenKE).

### Bước A — Chuẩn bị

1. Tạo notebook, bật **GPU**, bật **Internet** (cài package + tải SentenceTransformer lần đầu).
2. Gắn input (ví dụ `/kaggle/input/your-bundle/`) hoặc `git clone` hai repo của bạn lên GitHub rồi clone vào `working`.

### Bước B — Export OpenKE → GOLD

```bash
cd /kaggle/working/OpenKE-OpenKE-PyTorch
pip install -q torch  # thường đã có sẵn trên Kaggle

python scripts/gold_openke/openke_to_gold.py \
  --openke_dir /kaggle/working/data/FB15K237 \
  --out_dir /kaggle/working/gold_data/FB15K237_openke \
  --dataset_tag FB15K237_openke \
  --noise_ratio 0.05 \
  --rule_top_k 100 \
  --seed 42
```

(`data/FB15K237` là bản đủ `entity2id`, `relation2id`, `train2id`, …)

### Bước C — Cài GOLD và chạy `gold.py`

```bash
cd /kaggle/working/GOLD-main
pip install -r requirements.txt

python gold.py \
  --dataset FB15K237_openke \
  --dataset_path /kaggle/working/gold_data/FB15K237_openke \
  --model_name kaggle_run1 \
  --epoch 10 \
  --batch_size 128 \
  --topk 100 \
  --ptlm_model sentence-transformers/all-MiniLM-L6-v2 \
  --lr 0.001 \
  --local_lambda 0.1 \
  --global_lambda 0.01 \
  --neg_cnt 1 \
  --seed 5 \
  --output_tsv \
  --device auto
```

TSV tạo tại: `/kaggle/working/GOLD-main/FB15K237_openke_kaggle_run1.tsv` (tiền tố `{dataset}_{model_name}`).

Giới hạn VRAM: giảm `--batch_size` (ví dụ 64). Lần chạy đầu sẽ encode cả graph → có thể mất vài chục phút.

### Bước D — TSV → `train2id` đã lọc

```bash
cd /kaggle/working/OpenKE-OpenKE-PyTorch
python scripts/gold_openke/gold_scores_to_openke_train.py \
  --gold_tsv /kaggle/working/GOLD-main/FB15K237_openke_kaggle_run1.tsv \
  --openke_dir /kaggle/working/data/FB15K237 \
  --out_train2id /kaggle/working/data/FB15K237/train2id_gold_clean.txt \
  --drop_top_fraction 0.05
```

### Bước E — Build OpenKE (`Base.so`) và train TransE

Trên Kaggle (Linux):

```bash
cd /kaggle/working/OpenKE-OpenKE-PyTorch/openke
apt-get -qq update && apt-get -qq install -y build-essential
bash make.sh
cd ..
```

Trong script train (ví dụ sửa từ `examples/train_transe_FB15K237.py`): `in_path` trỏ tới `FB15K237`, dùng `tri_file=.../train2id_gold_clean.txt` như mục 5; `TestDataLoader` vẫn dùng cùng thư mục benchmark. Chạy train; **Output** Kaggle: lưu checkpoint / zip `train2id_gold_clean.txt` nếu cần giữ lâu.

### Lưu ý session Kaggle

- Hết giờ là mất `/kaggle/working` (trừ bản đã commit/submit output).
- Có thể tách notebook: (1) chỉ GOLD + lưu TSV + `train2id_gold_clean` lên Output/Dataset, (2) notebook sau chỉ train OpenKE.

---

## Patch đã áp dụng trong repo GOLD của bạn

Đường dẫn: `...\GOLD-main\gold.py`, `...\GOLD-main\data_loader.py`

- Dataset tùy chọn + `--dataset_path`, `--device auto|cpu|cuda`
- Tránh `batch_size * 0` khi không có GPU
- `torch.load(..., map_location=...)`
- Metrics an toàn khi `nr_error == 0`
- Rules: nếu thiếu file `.pkl`, tự dùng rules rỗng
- Errors: nếu thiếu file error, cảnh báo và trả danh sách rỗng
