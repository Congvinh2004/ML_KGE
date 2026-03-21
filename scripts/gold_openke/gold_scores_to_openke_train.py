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
from typing import Dict, List, Tuple


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
        help="Optional. If set, only keep triples whose (h,r,t) strings appear in this file. "
        "GOLD ConceptNet train.txt format: relation\\thead\\ttail (same as dataset/conceptnet/train.txt).",
    )
    args = ap.parse_args()

    if not 0.0 <= args.drop_top_fraction < 1.0:
        raise ValueError("drop_top_fraction must be in [0, 1)")

    ent2id, rel2id = load_openke_maps(os.path.abspath(args.openke_dir))

    allowed_hrt: set | None = None
    if args.restrict_train_triples_txt:
        allowed_hrt = set()
        with open(args.restrict_train_triples_txt, "r", encoding="utf-8") as tf:
            for line in tf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    raise ValueError(
                        "restrict file: need 3 cols r\\th\\tt (ConceptNet): %r" % line
                    )
                r_s, h_s, t_s = parts[0].strip(), parts[1].strip(), parts[2].strip()
                # Khớp cột TSV gold.py: h_str, r_str, t_str
                allowed_hrt.add((h_s, r_s, t_s))

    rows: List[Tuple[int, str, str, str, float]] = []
    with open(args.gold_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                raise ValueError("Bad TSV line (need 8 cols): %r" % line)
            label = int(parts[6])
            if label != 0 and not args.include_labeled_noise:
                continue
            h_s, r_s, t_s = parts[3], parts[4], parts[5]
            score = float(parts[7])
            if allowed_hrt is not None and (h_s, r_s, t_s) not in allowed_hrt:
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

    out_path = os.path.abspath(args.out_train2id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("%d\n" % len(triples_out))
        for h, t, r in triples_out:
            f.write("%d %d %d\n" % (h, t, r))

    print("✅ Wrote", out_path)
    print("   kept triples:", len(triples_out))
    print("   dropped (score > %.6f):" % threshold, dropped)
    if missing:
        print("   ⚠️  skipped (missing in entity2id/relation2id):", missing)


if __name__ == "__main__":
    main()
