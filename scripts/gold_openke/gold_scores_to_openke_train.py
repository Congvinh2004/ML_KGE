#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert GOLD test TSV (--output_tsv) -> OpenKE train2id file.

TSV columns (from official gold.py):
  h_id \\t r_id \\t t_id \\t h_str \\t r_str \\t t_str \\t label \\t score

Higher score == higher anomaly score in GOLD (same ordering as training eval).
We drop the top --drop_top_fraction fraction of triples by score (most suspicious).

Only triples with label 0 (clean graph triples) are considered for the new train set
by default, so injected synthetic errors in errors/*.txt are not written into train2id.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple


def read_openke_train2id_count(train2id_path: str) -> Optional[int]:
    """Dòng 1 định dạng OpenKE train2id: số triple."""
    if not os.path.isfile(train2id_path):
        return None
    with open(train2id_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    try:
        return int(first)
    except ValueError:
        return None


def load_openke_maps(
    openke_dir: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    def load_map(path: str) -> Dict[str, int]:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        n = int(lines[0].strip())
        m: Dict[str, int] = {}
        for line in lines[1 : 1 + n]:
            name, i = line.split("\t", 1)
            m[name.strip()] = int(i.strip())
        return m

    ent = load_map(os.path.join(openke_dir, "entity2id.txt"))
    rel = load_map(os.path.join(openke_dir, "relation2id.txt"))
    return ent, rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_tsv", required=True, help="GOLD output TSV path")
    ap.add_argument(
        "--openke_dir", required=True, help="OpenKE folder with entity2id/relation2id"
    )
    ap.add_argument("--out_train2id", required=True, help="Output train2id path")
    ap.add_argument(
        "--drop_top_fraction",
        type=float,
        default=0.05,
        help="Remove this fraction of triples with highest scores (0..1).",
    )
    ap.add_argument(
        "--include_labeled_noise",
        action="store_true",
        help="If set, also keep lines with label 1 (usually not wanted for training).",
    )
    ap.add_argument(
        "--restrict_train_triples_txt",
        default=None,
        help="Optional. If set, only keep triples whose (h,r,t) strings appear in this file "
        "(train split only — avoids valid/test leakage into OpenKE train).",
    )
    ap.add_argument(
        "--restrict_triple_format",
        choices=("hrt", "rht"),
        default="hrt",
        help="Column order in restrict file: hrt=head\\trelation\\ttail (openke_to_gold / WN18RR); "
        "rht=relation\\thead\\ttail (ConceptNet).",
    )
    args = ap.parse_args()

    if not 0.0 <= args.drop_top_fraction < 1.0:
        raise ValueError("drop_top_fraction must be in [0, 1)")

    ent2id, rel2id = load_openke_maps(os.path.abspath(args.openke_dir))

    allowed_hrt: set | None = None
    n_restrict_file = 0
    if args.restrict_train_triples_txt:
        allowed_hrt = set()
        with open(args.restrict_train_triples_txt, "r", encoding="utf-8") as tf:
            for line in tf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    raise ValueError("restrict file: need 3 tab-separated cols: %r" % line)
                if args.restrict_triple_format == "hrt":
                    h_s, r_s, t_s = (
                        parts[0].strip(),
                        parts[1].strip(),
                        parts[2].strip(),
                    )
                else:
                    r_s, h_s, t_s = (
                        parts[0].strip(),
                        parts[1].strip(),
                        parts[2].strip(),
                    )
                allowed_hrt.add((h_s, r_s, t_s))
                n_restrict_file += 1

    rows: List[Tuple[int, str, str, str, float]] = []
    tsv_nonempty_lines = 0
    tsv_skipped_label_nonzero = 0
    tsv_skipped_restrict = 0
    with open(args.gold_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tsv_nonempty_lines += 1
            parts = line.split("\t")
            if len(parts) < 8:
                raise ValueError("Bad TSV line (need 8 cols): %r" % line)
            label = int(parts[6])
            if label != 0 and not args.include_labeled_noise:
                tsv_skipped_label_nonzero += 1
                continue
            h_s, r_s, t_s = parts[3], parts[4], parts[5]
            score = float(parts[7])
            if allowed_hrt is not None and (h_s, r_s, t_s) not in allowed_hrt:
                tsv_skipped_restrict += 1
                continue
            rows.append((label, h_s, r_s, t_s, score))

    if not rows:
        raise RuntimeError("No rows read (check TSV path and label filter).")

    scores = [r[4] for r in rows]
    scores_sorted = sorted(scores)
    n = len(scores_sorted)

    # Giữ phần có score thấp nhất; bỏ top drop_top_fraction điểm cao nhất.
    if args.drop_top_fraction <= 0 or n == 0:
        threshold = float("inf")
    else:
        cut = n - int(n * args.drop_top_fraction)
        cut = max(1, min(cut, n))
        threshold = scores_sorted[cut - 1]

    kept: List[Tuple[str, str, str]] = []
    dropped = 0
    for _lb, h_s, r_s, t_s, sc in rows:
        if sc > threshold:
            dropped += 1
            continue
        kept.append((h_s, r_s, t_s))

    triples_out: List[Tuple[int, int, int]] = []
    missing = 0
    for h_s, r_s, t_s in kept:
        if h_s not in ent2id or t_s not in ent2id or r_s not in rel2id:
            missing += 1
            continue
        h, t, r = ent2id[h_s], ent2id[t_s], rel2id[r_s]
        triples_out.append((h, t, r))

    if not triples_out:
        raise RuntimeError(
            "train2id sau lọc rỗng (không còn triple). Kiểm tra TSV, drop_top_fraction, "
            "restrict_train_triples_txt, và cảnh báo missing entity/relation ở trên."
        )

    out_path = os.path.abspath(args.out_train2id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("%d\n" % len(triples_out))
        for h, t, r in triples_out:
            f.write("%d %d %d\n" % (h, t, r))

    openke_dir_abs = os.path.abspath(args.openke_dir)
    train2id_path = os.path.join(openke_dir_abs, "train2id.txt")
    n_openke_train = read_openke_train2id_count(train2id_path)
    n_final = len(triples_out)
    n_rows_label0 = len(rows)
    kept_after_score = len(kept)

    print("✅ Wrote", out_path)
    print("")
    print("=== Thống kê gold_scores_to_openke_train ===")
    print("openke_dir:", openke_dir_abs)
    print("gold_tsv:", os.path.abspath(args.gold_tsv))
    print("drop_top_fraction:", args.drop_top_fraction)
    if args.restrict_train_triples_txt:
        print(
            "restrict_train_triples_txt:",
            os.path.abspath(args.restrict_train_triples_txt),
        )
        print("restrict_triple_format:", args.restrict_triple_format)
        print("  Số dòng trong file restrict (train only):", n_restrict_file)
    if n_openke_train is not None:
        print("Số triple train gốc (train2id.txt, dòng 1):", n_openke_train)
    else:
        print("Số triple train gốc (train2id.txt): N/A (không đọc được file)")
    print("--- Đọc TSV ---")
    print("  Dòng TSV không rỗng:", tsv_nonempty_lines)
    print("  Bỏ qua (label != 0, không include_labeled_noise):", tsv_skipped_label_nonzero)
    if allowed_hrt is not None:
        print("  Bỏ qua (không thuộc restrict_train_triples_txt):", tsv_skipped_restrict)
    print("  Số triple (label 0) đưa vào bước lọc theo score:", n_rows_label0)
    print("--- Lọc theo score (GOLD) ---")
    if args.drop_top_fraction <= 0:
        print("  Không lọc theo score (drop_top_fraction <= 0).")
    else:
        print("  Ngưỡng score (giữ score <= threshold):", threshold)
    print("  Loại vì score cao (dropped):", dropped)
    print("  Còn lại sau lọc score (chuỗi h,r,t):", kept_after_score)
    print("--- Ánh xạ sang id OpenKE ---")
    print("  Bỏ vì thiếu entity/relation trong map (missing):", missing)
    print("  Số triple ghi vào train2id output:", n_final)
    if n_openke_train is not None:
        removed_vs_orig = n_openke_train - n_final
        pct = (100.0 * removed_vs_orig / n_openke_train) if n_openke_train else 0.0
        print("--- So với train2id.txt gốc ---")
        if allowed_hrt is not None and n_rows_label0 != n_openke_train:
            print(
                "  Lưu ý: |TSV label-0 sau restrict| = %d khác |train2id.txt| = %d — "
                "kiểm tra train.txt / TSV thiếu triple train."
                % (n_rows_label0, n_openke_train)
            )
        elif allowed_hrt is None and n_rows_label0 != n_openke_train:
            print(
                "  Lưu ý: |TSV label-0| = %d khác |train2id.txt| = %d — "
                "dùng --restrict_train_triples_txt (train.txt) để chỉ lọc split train."
                % (n_rows_label0, n_openke_train)
            )
        elif allowed_hrt is not None:
            print(
                "  Chế độ train-only: lọc score trên %d triple train (không valid/test)."
                % n_rows_label0
            )
        print("  Đã không đưa vào file train sạch (tổng cộng):", removed_vs_orig)
        print("  (gồm: không có trong TSV label-0, bị drop score, missing map, v.v.)")
        print("  Tỷ lệ so với train gốc: %.4f%%" % pct)
    print("=== Hết thống kê ===")
    print("")


if __name__ == "__main__":
    main()
