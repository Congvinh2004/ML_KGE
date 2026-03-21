# Hướng dẫn từng cell — chạy GOLD với ConceptNet (bộ gốc trong repo GOLD)

Tài liệu này song song với [`kaggle_gold_openke_pipeline.ipynb`](kaggle_gold_openke_pipeline.ipynb) (FB15K237 → OpenKE), nhưng **ConceptNet** đã được tác giả GOLD đóng gói sẵn trong `GOLD-main/dataset/conceptnet/`. Bạn **không** chạy `openke_to_gold.py` và **không** cần bước TransE/OpenKE ở cuối trừ khi tự mở rộng ngoài paper.

---

## Chuẩn bị (trước “Cell 1”)

1. **Repo:** clone hoặc copy [GOLD-main](https://github.com/...) — thư mục làm việc gọi là `GOLD_ROOT` (ví dụ `.../GOLD-main`).
2. **Dữ liệu ConceptNet** nằm tại:
   - `GOLD_ROOT/dataset/conceptnet/train.txt`, `valid.txt`, `test.txt`
   - `GOLD_ROOT/dataset/conceptnet/errors/` — ví dụ `C-05-error.txt`, `C-10-error.txt`, `C-20-error.txt` (tỷ lệ nhiễu 5% / 10% / 20%).
   - `GOLD_ROOT/dataset/conceptnet/rules/` — ví dụ `C-05-rules-top-100.pkl` (luật AMIE đã xử lý; `--topk` phải khớp tên file).
3. **Kaggle (tuỳ chọn):** Add Input dataset chứa `GOLD-main`, hoặc `git clone` + bật Internet để `pip install` và tải SentenceTransformer.

Mọi lệnh `python gold.py` dưới đây **chạy với thư mục hiện tại = `GOLD_ROOT`** (`cd` vào đó trước), vì code dùng đường dẫn tương đối `./dataset/conceptnet`.

---

## Cell 0 — Giới thiệu (Markdown)

**Nội dung:** Pipeline ConceptNet = huấn luyện + đánh giá nhị phân (triple đúng / có lỗi) trong GOLD; file kết quả TSV (nếu bật) là `conceptnet_<model_name>.tsv` — **không** phải `C-05_<model_name>.tsv`.

---

## Cell 1 — Cấu hình đường dẫn và tham số

Chạy một cell Python đặt biến (chỉnh theo máy bạn):

| Biến | Ý nghĩa |
|------|--------|
| `WORK` | Thư mục làm việc, ví dụ `/kaggle/working` hoặc thư mục project |
| `GOLD_ROOT` | `os.path.join(WORK, "GOLD-main")` hoặc đường dẫn tuyệt đối tới repo GOLD |
| `DATASET` | Một trong: **`C-05`**, **`C-10`**, **`C-20`** — khớp với file `errors/<DATASET>-error.txt` và `rules/<DATASET>-rules-top-<K>.pkl` |
| `MODEL_NAME` | Tên run, ví dụ `train`, `exp1` — dùng trong log và tên file TSV |
| `RULE_TOP_K` / `--topk` | Phải trùng số trong tên file rules, thường **100** trong repo gốc |

**Lưu ý:** Không đặt `--dataset_path` khi dùng ConceptNet chuẩn — GOLD tự trỏ tới `./dataset/conceptnet`.

---

## Cell 2 — Copy hoặc clone repo GOLD vào `working`

- **Kaggle:** Add dataset có `GOLD-main`, hoặc:
  - `git clone --depth 1 <URL_GOLD> GOLD_ROOT`
- **Local:** đảm bảo `GOLD_ROOT/gold.py` và `GOLD_ROOT/dataset/conceptnet/` tồn tại.

Kiểm tra nhanh:

```text
GOLD_ROOT/gold.py
GOLD_ROOT/dataset/conceptnet/train.txt
GOLD_ROOT/dataset/conceptnet/errors/C-05-error.txt   (nếu chọn C-05)
GOLD_ROOT/dataset/conceptnet/rules/C-05-rules-top-100.pkl   (topk=100)
```

Khác với notebook FB15K: **không** cần `OpenKE-OpenKE-PyTorch`, **không** cần `benchmarks/FB15K237`.

---

## Cell 3 — (Bỏ qua so với FB15K) Không export OpenKE → GOLD

Trong pipeline FB15K, cell này chạy `openke_to_gold.py`. Với ConceptNet **không chạy bước này** — dữ liệu đã đúng layout GOLD.

---

## Cell 4 — Cài dependency

Trong terminal hoặc cell notebook (từ `GOLD_ROOT` hoặc dùng `pip` global):

```bash
cd GOLD_ROOT
pip install -r requirements.txt
```

Cần mạng lần đầu để tải weights `sentence-transformers` (theo `--ptlm_model`).

---

## Cell 5 — Huấn luyện và test GOLD (`gold.py`)

Ví dụ **C-05**, batch nhỏ nếu GPU yếu:

```bash
cd GOLD_ROOT
python gold.py ^
  --dataset C-05 ^
  --model_name kaggle_cn1 ^
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

**Tham số quan trọng:**

- `--dataset`: `C-05` / `C-10` / `C-20` — quyết định file `errors/` và `rules/` dùng tập nhiễu tương ứng.
- `--topk`: trùng với file `*-rules-top-<K>.pkl` (repo gốc thường **100**).
- `--ptlm_model`: mặc định trong README GOLD là `sentence-t5-xxl` (nặng); `all-MiniLM-L6-v2` nhẹ hơn cho máy yếu hoặc thử nhanh.

**Đầu ra:**

- Log: `GOLD_ROOT/log/C-05_<model_name>_log.txt` (tên file phụ thuộc `--dataset` và `--model_name`).
- Nếu có `--output_tsv`: file **`conceptnet_<model_name>.tsv`** nằm **trong `GOLD_ROOT`** (thư mục hiện tại khi chạy lệnh), ví dụ `conceptnet_kaggle_cn1.tsv`.

Trên console sẽ có kết quả kiểu **Acc**, **AUC** (test).

---

## Cell 6 — (Bỏ qua) TSV → `train2id_gold_clean.txt`

Bước `gold_scores_to_openke_train.py` chỉ dành cho **chuỗi OpenKE** (FB15K237, …). Với ConceptNet bạn **không** có `train2id.txt` kiểu OpenKE — **bỏ qua cell này**.

Nếu cần phân tích điểm từng triple, dùng trực tiếp file `conceptnet_<model_name>.tsv` từ Cell 5.

---

## Cell 7 — (Bỏ qua) `make.sh` / OpenKE `Base.so`

Chỉ cần khi huấn luyện **TransE** trong OpenKE. Pipeline ConceptNet trong paper **không** gồm bước này.

---

## Cell 8 — (Bỏ qua) Train TransE

Không áp dụng cho benchmark ConceptNet gốc GOLD.

---

## Cell 9 — (Tuỳ chọn) Đóng gói artifact (Kaggle)

Nếu chạy trên Kaggle, có thể nén `conceptnet_<model_name>.tsv`, thư mục `log/`, checkpoint nếu bạn lưu thêm — không bắt buộc.

---

## Tóm tắt mapping notebook FB15K → ConceptNet

| Notebook FB15K | ConceptNet (GOLD gốc) |
|----------------|------------------------|
| Cell 1–2: OpenKE + GOLD + FB15K237 | Chỉ cần **GOLD-main** + dataset conceptnet |
| Cell 3: `openke_to_gold.py` | **Không dùng** |
| Cell 4: pip | `pip install -r requirements.txt` trong GOLD |
| Cell 5: `gold.py` + `--dataset_path` | `gold.py` + `--dataset C-05` (không `--dataset_path`) |
| Cell 6–8: lọc train + TransE | **Không dùng** |

---

## Gợi ý thêm (tuỳ đề tài)

1. **Tự khai phá luật mới:** xem `GOLD-main/scripts/process_amie_result.py` và README GOLD (AMIE) — cần output AMIE và `--dataset C-05` khớp khi post-process.
2. **So sánh mức nhiễu:** chạy lần lượt `C-05`, `C-10`, `C-20` và so sánh AUC/Acc trong log.

---

## Một cell Python gọn (copy vào notebook ConceptNet)

```python
import os, subprocess, sys

WORK = "/kaggle/working"  # hoặc thư mục local
GOLD_ROOT = os.path.join(WORK, "GOLD-main")
os.chdir(GOLD_ROOT)

DATASET = "C-05"       # C-10, C-20
MODEL_NAME = "run_cn1"
TOPK = 100

cmd = [
    sys.executable, "gold.py",
    "--dataset", DATASET,
    "--model_name", MODEL_NAME,
    "--epoch", "10",
    "--batch_size", "256",
    "--topk", str(TOPK),
    "--ptlm_model", "sentence-transformers/all-MiniLM-L6-v2",
    "--lr", "0.001",
    "--local_lambda", "0.1",
    "--global_lambda", "0.01",
    "--neg_cnt", "1",
    "--seed", "5",
    "--output_tsv",
    "--device", "auto",
]
print("$", " ".join(cmd))
subprocess.run(cmd, check=True, cwd=GOLD_ROOT)
print("TSV (nếu có):", os.path.join(GOLD_ROOT, f"conceptnet_{MODEL_NAME}.tsv"))
```

Đặt `WORK` / `GOLD_ROOT` đúng chỗ clone repo trước khi chạy.

---

## Kết hợp thêm TransE (ConceptNet → OpenKE → GOLD → lọc → KGE)

1. **`conceptnet_to_openke.py`** — xuất `dataset/conceptnet/` sang thư mục OpenKE (khớp id entity/relation với GOLD).
2. Chạy **`gold.py`** như trên → TSV `conceptnet_<model>.tsv`.
3. **`gold_scores_to_openke_train.py`** với **`--restrict_train_triples_txt`** trỏ tới `dataset/conceptnet/train.txt` — chỉ giữ triple **tập train** sau khi lọc điểm (tránh lẫn valid/test vào train KGE).
4. Notebook **`kaggle_conceptnet_gold_openke_pipeline.ipynb`** gói đủ các bước trên + `make.sh` + TransE.
