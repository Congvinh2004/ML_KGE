#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chạy TransE + link prediction trong process riêng (tránh OOM kernel Jupyter trên Kaggle Save Version).

Ví dụ:
  python run_transe_conceptnet_openke.py \\
    --openke_root /path/to/OpenKE-OpenKE-PyTorch \\
    --data_path /path/to/ConceptNet_openke \\
    --tri_file /path/to/train2id_gold_clean.txt \\
    --ckpt_out /path/to/transe_cn_gold.ckpt
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys


def main() -> None:
    print("run_transe_conceptnet_openke: start", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--openke_root", required=True, help="Repo OpenKE (gốc có thư mục openke/)")
    ap.add_argument(
        "--data_path",
        required=True,
        help="Thư mục OpenKE benchmark (entity2id, valid2id, test2id, ...)",
    )
    ap.add_argument("--tri_file", required=True, help="train2id đã lọc GOLD")
    ap.add_argument("--train_times", type=int, default=300)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--nbatches", type=int, default=200)
    ap.add_argument("--neg_ent", type=int, default=5)
    ap.add_argument("--ckpt_out", required=True, help="File .ckpt để lưu")
    ap.add_argument(
        "--metrics_out",
        default=None,
        help="Nếu set, ghi MRR/MR/Hits@k ra file .txt (link prediction)",
    )
    ap.add_argument("--margin", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument(
        "--pin_memory",
        action="store_true",
        help="Bật pin_memory cho DataLoader (mặc định tắt — giảm RAM trên Kaggle / ConceptNet lớn).",
    )
    ap.add_argument(
        "--use_cpu",
        action="store_true",
        help="Ép huấn luyện + eval trên CPU (chậm nhưng tránh OOM GPU).",
    )
    args = ap.parse_args()
    print("run_transe_conceptnet_openke: args OK", flush=True)

    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    root = os.path.abspath(args.openke_root)
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    data_path = os.path.abspath(args.data_path).replace("\\", "/") + "/"

    import torch
    from openke.config import Trainer, Tester
    from openke.data import TestDataLoader
    from openke.data.PyTorchTrainDataLoader import PyTorchTrainDataLoader
    from openke.module.loss import MarginLoss
    from openke.module.model import TransE
    from openke.module.strategy import NegativeSampling

    tri_file = os.path.abspath(args.tri_file)
    use_gpu = torch.cuda.is_available()
    print("USE_GPU:", use_gpu)
    if use_gpu:
        torch.cuda.empty_cache()

    _dl_kw = dict(
        in_path=data_path,
        tri_file=tri_file,
        nbatches=args.nbatches,
        threads=0,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_ent,
        neg_rel=0,
    )
    # Repo cũ không có pin_memory → TypeError → gọi lại không truyền pin_memory.
    try:
        train_dataloader = PyTorchTrainDataLoader(**_dl_kw, pin_memory=args.pin_memory)
    except TypeError as e:
        if "pin_memory" in str(e):
            train_dataloader = PyTorchTrainDataLoader(**_dl_kw)
        else:
            raise
    _bs = train_dataloader.get_batch_size()
    _nt = train_dataloader.get_triple_tot()
    if _bs < 1:
        print(
            "Lỗi: batch_size=%s — --nbatches (%d) > số triples train (%d). Giảm --nbatches."
            % (_bs, args.nbatches, _nt),
            file=sys.stderr,
        )
        sys.exit(1)
    test_dataloader = TestDataLoader(data_path, "link")

    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        p_norm=1,
        norm_flag=True,
    )
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=args.margin),
        batch_size=train_dataloader.get_batch_size(),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_out)) or ".", exist_ok=True)
    ckpt_path = os.path.abspath(args.ckpt_out)

    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.train_times,
        alpha=args.alpha,
        use_gpu=use_gpu,
    )
    trainer.run()
    transe.save_checkpoint(ckpt_path)
    print("Saved:", ckpt_path)

    transe.load_checkpoint(ckpt_path)
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=use_gpu)
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=True)

    if args.metrics_out:
        lines = [
            "TransE link prediction (filtered train, type_constrain=True)",
            "time_utc: %s" % datetime.datetime.utcnow().isoformat() + "Z",
            "openke_root: %s" % root,
            "data_path: %s" % data_path,
            "tri_file: %s" % tri_file,
            "train_times: %d" % args.train_times,
            "dim: %d  nbatches: %d  neg_ent: %d" % (args.dim, args.nbatches, args.neg_ent),
            "ckpt: %s" % ckpt_path,
            "",
            "MRR\t%f" % mrr,
            "MR\t%f" % mr,
            "Hits@10\t%f" % hit10,
            "Hits@3\t%f" % hit3,
            "Hits@1\t%f" % hit1,
            "",
        ]
        mp = os.path.abspath(args.metrics_out)
        os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
        with open(mp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Wrote metrics:", mp)


if __name__ == "__main__":
    # In ra stderr ngay (không buffer) — nếu log không có dòng này, tiến trình chết trước khi chạy Python hoặc repo clone chưa đúng bản.
    sys.stderr.write("run_transe_conceptnet_openke: boot (stderr)\n")
    sys.stderr.flush()
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)
