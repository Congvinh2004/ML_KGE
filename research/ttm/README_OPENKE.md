# 📚 OpenKE: An Open Toolkit for Knowledge Embedding

## 📋 Thông tin tổng quan

**OpenKE** là một bộ công cụ mã nguồn mở được thiết kế để hỗ trợ việc nhúng tri thức (Knowledge Embedding) trong các đồ thị tri thức lớn. Bộ công cụ này cung cấp các mô hình và thuật toán hiệu quả để biểu diễn các thực thể và quan hệ trong không gian vector, giúp cải thiện hiệu suất của các tác vụ như suy luận, tìm kiếm và hoàn thiện tri thức.

---

## 📄 Thông tin bài báo

### Paper chính

- **Tiêu đề:** OpenKE: An Open Toolkit for Knowledge Embedding
- **Tác giả:** 
  - Xu Han
  - Shulin Cao
  - Xin Lv
  - Yankai Lin
  - Zhiyuan Liu
  - Maosong Sun
  - Juanzi Li
- **Tổ chức:** THUNLP (Tsinghua Natural Language Processing Group)
- **Hội nghị:** EMNLP 2018 (Empirical Methods in Natural Language Processing)
- **Track:** Demo Track
- **Năm xuất bản:** 2018

### Link bài báo

- **ACL Anthology:** https://aclanthology.org/D18-2024/
- **PDF trực tiếp:** https://aclanthology.org/D18-2024.pdf

### Uy tín hội nghị

- **CORE Ranking:** A* (Hàng đầu)
- **Vị trí:** Top 3 hội nghị NLP (cùng với ACL và NAACL)
- **Impact Score:** 21.90 (Research.com)
- **Acceptance Rate:** ~25% (chất lượng cao)

---

## 🔗 Link GitHub

- **Repository chính:** https://github.com/thunlp/OpenKE
- **Branch PyTorch:** https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch
- **Website:** http://openke.thunlp.org/

---

## ⭐ Tính năng chính

### 1. Hỗ trợ nhiều mô hình Knowledge Embedding

OpenKE triển khai các mô hình phổ biến:

#### Translational Distance Models
- **TransE** - Translating Embeddings for Modeling Multi-relational Data
- **TransH** - Knowledge Graph Embedding by Translating on Hyperplanes
- **TransR** - Learning Entity and Relation Embeddings for Knowledge Graph Completion
- **TransD** - Knowledge Graph Embedding via Dynamic Mapping Matrix

#### Semantic Matching Models
- **RESCAL** - A Three-Way Model for Collective Learning on Multi-Relational Data
- **DistMult** - Embedding Entities and Relations for Learning and Inference in Knowledge Bases
- **ComplEx** - Complex Embeddings for Simple Link Prediction
- **Analogy** - Analogical Inference for Multi-relational Embeddings

### 2. Hiệu suất cao

- **C++ backend:** Sử dụng C++ cho các thao tác cơ bản (negative sampling, scoring)
- **PyTorch integration:** Sử dụng PyTorch cho các mô hình cụ thể
- **GPU support:** Tối ưu hóa việc huấn luyện trên GPU
- **Multi-threading:** Hỗ trợ đa luồng cho xử lý song song

### 3. Dễ dàng mở rộng

- **Modular design:** Thiết kế module, dễ dàng tích hợp mô hình mới
- **Flexible API:** API linh hoạt cho training và evaluation
- **Customizable:** Cho phép tùy chỉnh loss functions, sampling strategies

### 4. Datasets được hỗ trợ

- **FB15K** - Freebase subset (14,951 entities, 1,345 relations)
- **FB15K237** - Filtered version của FB15K (14,541 entities, 237 relations)
- **WN18** - WordNet subset (40,943 entities, 18 relations)
- **WN18RR** - Filtered version của WN18 (40,943 entities, 11 relations)
- **YAGO3-10** - YAGO subset
- **DBpedia** - DBpedia subset

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống

- **Python:** 3.6+
- **PyTorch:** 1.0+
- **CUDA:** (Optional) Cho GPU support
- **GCC:** Cho việc compile C++ code

### Cài đặt

#### 1. Clone repository

```bash
# Clone branch PyTorch
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE.git
cd OpenKE
```

#### 2. Build native library

```bash
cd openke
bash make.sh
# Hoặc trên Windows:
# make.bat
```

#### 3. Kiểm tra cài đặt

```bash
cd ..
python -c "import openke; print('OpenKE installed successfully!')"
```

### Cài đặt từ pip (nếu có)

```bash
pip install openke
```

---

## 📖 Hướng dẫn sử dụng

### Ví dụ cơ bản: Train TransE trên FB15K237

```python
import sys
sys.path.append('.')

from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# 1. Load data
train_dataloader = TrainDataLoader(
    in_path = "./benchmarks/FB15K237/",
    nbatches = 100,
    threads = 8,
    sampling_mode = "normal",
    bern_flag = 1,
    filter_flag = 1,
    neg_ent = 25,
    neg_rel = 0
)

test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# 2. Define model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 200,
    p_norm = 1,
    norm_flag = True
)

# 3. Define loss function
model = NegativeSampling(
    model = transe,
    loss = MarginLoss(margin = 5.0),
    batch_size = train_dataloader.get_batch_size()
)

# 4. Train
trainer = Trainer(
    model = model,
    data_loader = train_dataloader,
    train_times = 1000,
    alpha = 0.5,
    use_gpu = True
)
trainer.run()

# 5. Test
transe.save_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
```

---

## 📁 Định dạng dữ liệu

OpenKE yêu cầu các file sau trong thư mục dataset:

### Files bắt buộc

1. **entity2id.txt**
   ```
   [số lượng entities]
   entity_name_1   0
   entity_name_2   1
   ...
   ```

2. **relation2id.txt**
   ```
   [số lượng relations]
   relation_name_1   0
   relation_name_2   1
   ...
   ```

3. **train2id.txt**
   ```
   [số lượng triples]
   head_id    tail_id    relation_id
   ...
   ```

4. **test2id.txt** (cho evaluation)
   ```
   [số lượng triples]
   head_id    tail_id    relation_id
   ...
   ```

5. **valid2id.txt** (cho validation, optional)
   ```
   [số lượng triples]
   head_id    tail_id    relation_id
   ...
   ```

### Files tùy chọn

6. **type_constrain.txt** (cho type constraint)
   ```
   [relation_id]    [số lượng head types]    [số lượng tail types]
   head_type_1    head_type_2    ...
   tail_type_1    tail_type_2    ...
   ...
   ```

---

## 📊 Kết quả benchmark

### FB15K237

| Model | MRR | Hits@10 | Hits@3 | Hits@1 |
|-------|-----|---------|--------|--------|
| TransE | 0.290 | 0.489 | 0.320 | 0.192 |
| TransH | 0.294 | 0.497 | 0.323 | 0.198 |
| TransR | 0.301 | 0.509 | 0.331 | 0.204 |
| TransD | 0.303 | 0.512 | 0.333 | 0.206 |
| RESCAL | 0.291 | 0.495 | 0.321 | 0.195 |
| DistMult | 0.313 | 0.521 | 0.345 | 0.218 |
| ComplEx | 0.315 | 0.523 | 0.347 | 0.220 |

### WN18RR

| Model | MRR | Hits@10 | Hits@3 | Hits@1 |
|-------|-----|---------|--------|--------|
| TransE | 0.226 | 0.512 | 0.350 | 0.043 |
| TransH | 0.230 | 0.518 | 0.354 | 0.045 |
| TransR | 0.238 | 0.528 | 0.362 | 0.048 |
| TransD | 0.240 | 0.530 | 0.364 | 0.049 |
| RESCAL | 0.228 | 0.515 | 0.352 | 0.044 |
| DistMult | 0.243 | 0.541 | 0.375 | 0.051 |
| ComplEx | 0.245 | 0.543 | 0.377 | 0.052 |

*Lưu ý: Kết quả có thể khác nhau tùy theo hyperparameters và số epochs training.*

---

## 🔧 Cấu trúc thư mục

```
OpenKE/
├── openke/                    # Core library
│   ├── module/
│   │   ├── model/            # Các mô hình KGE
│   │   ├── loss/             # Loss functions
│   │   └── strategy/         # Training strategies
│   ├── data/                 # Data loaders
│   └── config/               # Configuration
├── benchmarks/               # Datasets
│   ├── FB15K237/
│   ├── WN18RR/
│   └── ...
├── examples/                 # Example scripts
│   ├── train_transe_FB15K237.py
│   └── ...
└── README.md
```

---

## 📝 Trích dẫn (Citation)

Nếu bạn sử dụng OpenKE trong nghiên cứu của mình, vui lòng trích dẫn:

### BibTeX

```bibtex
@inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv, Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
   pages={139--144},
   year={2018},
   address={Brussels, Belgium},
   publisher={Association for Computational Linguistics},
   url={https://aclanthology.org/D18-2024}
}
```

### APA Style

Han, X., Cao, S., Lv, X., Lin, Y., Liu, Z., Sun, M., & Li, J. (2018). OpenKE: An Open Toolkit for Knowledge Embedding. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 139-144). Association for Computational Linguistics.

---

## 👥 Tác giả và đóng góp

### Tác giả chính

- **Xu Han** - THUNLP, Tsinghua University
- **Shulin Cao** - THUNLP, Tsinghua University
- **Xin Lv** - THUNLP, Tsinghua University
- **Yankai Lin** - THUNLP, Tsinghua University
- **Zhiyuan Liu** - THUNLP, Tsinghua University
- **Maosong Sun** - THUNLP, Tsinghua University
- **Juanzi Li** - THUNLP, Tsinghua University

### Tổ chức

- **THUNLP** (Tsinghua Natural Language Processing Group)
- **Tsinghua University**, Beijing, China

### Đóng góp

OpenKE là một dự án mã nguồn mở. Chúng tôi hoan nghênh các đóng góp từ cộng đồng. Vui lòng xem [CONTRIBUTING.md](https://github.com/thunlp/OpenKE/blob/master/CONTRIBUTING.md) để biết thêm chi tiết.

---

## 📄 License

OpenKE được phát hành dưới giấy phép **MIT License**. Xem file [LICENSE](https://github.com/thunlp/OpenKE/blob/master/LICENSE) để biết thêm chi tiết.

---

## 🔗 Tài liệu tham khảo

### Papers liên quan

1. **TransE:** Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. *NIPS*.

2. **TransH:** Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). Knowledge graph embedding by translating on hyperplanes. *AAAI*.

3. **TransR:** Lin, Y., Liu, Z., Sun, M., Liu, Y., & Zhu, X. (2015). Learning entity and relation embeddings for knowledge graph completion. *AAAI*.

4. **TransD:** Ji, G., He, S., Xu, L., Liu, K., & Zhao, J. (2015). Knowledge graph embedding via dynamic mapping matrix. *ACL*.

5. **DistMult:** Yang, B., Yih, W. T., He, X., Gao, J., & Deng, L. (2015). Embedding entities and relations for learning and inference in knowledge bases. *ICLR*.

6. **ComplEx:** Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex embeddings for simple link prediction. *ICML*.

### Links hữu ích

- **GitHub Repository:** https://github.com/thunlp/OpenKE
- **ACL Anthology:** https://aclanthology.org/D18-2024/
- **THUNLP:** http://nlp.csai.tsinghua.edu.cn/
- **OpenKE Website:** http://openke.thunlp.org/

---

## ❓ FAQ (Frequently Asked Questions)

### Q: OpenKE có hỗ trợ GPU không?
**A:** Có, OpenKE hỗ trợ đầy đủ GPU training. Chỉ cần set `use_gpu = True` trong Trainer.

### Q: Làm sao để thêm mô hình mới vào OpenKE?
**A:** Tạo class mới kế thừa từ `openke.module.model.Model` và implement các methods cần thiết.

### Q: OpenKE có hỗ trợ custom loss function không?
**A:** Có, bạn có thể tạo custom loss function kế thừa từ `openke.module.loss.Loss`.

### Q: Làm sao để load checkpoint và tiếp tục training?
**A:** Sử dụng `model.load_checkpoint(path)` để load checkpoint đã lưu.

### Q: OpenKE có hỗ trợ multi-GPU không?
**A:** Hiện tại OpenKE chưa hỗ trợ multi-GPU training trực tiếp, nhưng có thể sử dụng PyTorch DataParallel.

---

## 📞 Liên hệ

- **Email:** openke@thunlp.org
- **GitHub Issues:** https://github.com/thunlp/OpenKE/issues
- **Website:** http://openke.thunlp.org/

---

**Last Updated:** 2024  
**Version:** OpenKE-PyTorch (Latest)

---

*Tài liệu này được tổng hợp từ bài báo gốc, GitHub repository và các tài liệu chính thức của OpenKE.*

























