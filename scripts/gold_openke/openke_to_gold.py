#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export OpenKE benchmark text files -> GOLD folder layout:
  <out_dir>/train.txt, valid.txt, test.txt   (each line: h\\tr\\tt string)
  <out_dir>/errors/<dataset_tag>-error.txt
  <out_dir>/rules/<dataset_tag>-rules-top-<K>.pkl   (empty rule sets per relation)

OpenKE train2id format (per line): head_id tail_id rel_id  (space-separated).
entity2id / relation2id: first line = count; then lines "name\\tid" (tab).
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Dict, List, Set, Tuple


def _read_count_and_lines(path: str) -> Tuple[int, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError("Empty file: %s" % path)
    n = int(lines[0].strip())
    body = lines[1:]
    if len(body) != n:
        print(
            "⚠️  Warning: %s header says %d lines but found %d data lines"
            % (path, n, len(body))
        )
    return n, body


def load_entity2id(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    _, lines = _read_count_and_lines(path)
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            raise ValueError("Bad entity2id line: %r" % line)
        name, eid_s = parts[0].strip(), parts[1].strip()
        eid = int(eid_s)
        id2name[eid] = name
        name2id[name] = eid
    return id2name, name2id


def load_relation2id(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    _, lines = _read_count_and_lines(path)
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            raise ValueError("Bad relation2id line: %r" % line)
        name, rid_s = parts[0].strip(), parts[1].strip()
        rid = int(rid_s)
        id2name[rid] = name
        name2id[name] = rid
    return id2name, name2id


def load_train2id_triples(path: str) -> List[Tuple[int, int, int]]:
    _, lines = _read_count_and_lines(path)
    triples: List[Tuple[int, int, int]] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError("Bad train2id line: %r" % line)
        h, t, r = map(int, parts)
        triples.append((h, t, r))
    return triples


def triple_to_line(
    h: int,
    t: int,
    r: int,
    ent: Dict[int, str],
    rel: Dict[int, str],
) -> str:
    return "%s\t%s\t%s" % (ent[h], rel[r], ent[t])


def write_triple_file(path: str, triples: List[Tuple[int, int, int]], ent, rel) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, t, r in triples:
            f.write(triple_to_line(h, t, r, ent, rel) + "\n")


def build_true_string_set(triples: List[Tuple[int, int, int]], ent, rel) -> Set[Tuple[str, str, str]]:
    s: Set[Tuple[str, str, str]] = set()
    for h, t, r in triples:
        s.add((ent[h], rel[r], ent[t]))
    return s


def _entity_ids_in_triples(triples: List[Tuple[int, int, int]]) -> List[int]:
    """Chỉ entity đã xuất hiện trong graph — GOLD Dataset.read() không có mọi id trong entity2id."""
    s: Set[int] = set()
    for h, t, r in triples:
        s.add(h)
        s.add(t)
    return list(s)


def sample_corrupted_triples(
    base_triples: List[Tuple[int, int, int]],
    ent: Dict[int, str],
    rel: Dict[int, str],
    true_str: Set[Tuple[str, str, str]],
    n_errors: int,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    if n_errors <= 0:
        return []
    ent_ids = _entity_ids_in_triples(base_triples)
    if len(ent_ids) < 2:
        print("⚠️  Too few entities in triples for corruption sampling; skipping errors.")
        return []
    out: List[Tuple[str, str, str]] = []
    attempts = 0
    max_attempts = max(1000, n_errors * 200)
    while len(out) < n_errors and attempts < max_attempts:
        attempts += 1
        h, t, r = rng.choice(base_triples)
        h_s, r_s, t_s = ent[h], rel[r], ent[t]
        if rng.random() < 0.5:
            nh = rng.choice(ent_ids)
            while nh == h:
                nh = rng.choice(ent_ids)
            h_s = ent[nh]
        else:
            nt = rng.choice(ent_ids)
            while nt == t:
                nt = rng.choice(ent_ids)
            t_s = ent[nt]
        cand = (h_s, r_s, t_s)
        if cand in true_str:
            continue
        out.append(cand)
    if len(out) < n_errors:
        print(
            "⚠️  Could only sample %d / %d unique corrupted triples "
            "(try larger graph or lower --noise_ratio)."
            % (len(out), n_errors)
        )
    return out


def write_errors(path: str, corrupted: List[Tuple[str, str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in corrupted:
            f.write("%s\t%s\t%s\n" % (h, r, t))


def write_empty_rules_pkl(path: str, relation_names: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rel2rules = {name: {} for name in relation_names}
    with open(path, "wb") as f:
        pickle.dump(rel2rules, f, protocol=pickle.HIGHEST_PROTOCOL)


def split_triples(
    triples: List[Tuple[int, int, int]],
    fr_train: float,
    fr_valid: float,
    rng: random.Random,
) -> Tuple[List, List, List]:
    idx = list(range(len(triples)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * fr_train)
    n_valid = int(n * fr_valid)
    i_tr = idx[:n_train]
    i_va = idx[n_train : n_train + n_valid]
    i_te = idx[n_train + n_valid :]
    if not i_te:
        i_te = i_va[-1:] if i_va else i_tr[-1:]
        i_va = i_va[:-1] if len(i_va) > 1 else i_va
    tr = [triples[i] for i in i_tr]
    va = [triples[i] for i in i_va]
    te = [triples[i] for i in i_te]
    return tr, va, te


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenKE -> GOLD dataset layout")
    ap.add_argument("--openke_dir", required=True, help="Folder with entity2id, relation2id, train2id")
    ap.add_argument("--out_dir", required=True, help="GOLD dataset folder (see GOLD dataset_path)")
    ap.add_argument(
        "--dataset_tag",
        required=True,
        help="Must match gold.py --dataset (used in errors/*.txt and rules/*.pkl names)",
    )
    ap.add_argument(
        "--noise_ratio",
        type=float,
        default=0.05,
        help="Fraction of |train| for synthetic error triples (GOLD training).",
    )
    ap.add_argument(
        "--rule_top_k",
        type=int,
        default=100,
        help="Must match gold.py --topk (filename rules/*-top-{k}.pkl).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split_train_only",
        action="store_true",
        help="If set, ignore valid2id/test2id and split train2id into train/valid/test.",
    )
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--valid_frac", type=float, default=0.1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    openke_dir = os.path.abspath(args.openke_dir)
    out_dir = os.path.abspath(args.out_dir)
    ent_path = os.path.join(openke_dir, "entity2id.txt")
    rel_path = os.path.join(openke_dir, "relation2id.txt")
    train_path = os.path.join(openke_dir, "train2id.txt")
    valid_path = os.path.join(openke_dir, "valid2id.txt")
    test_path = os.path.join(openke_dir, "test2id.txt")

    id2ent, _ = load_entity2id(ent_path)
    id2rel, _ = load_relation2id(rel_path)
    train_triples = load_train2id_triples(train_path)

    if args.split_train_only or not os.path.isfile(valid_path) or not os.path.isfile(test_path):
        print("📂 Building train/valid/test split from train2id only.")
        tr, va, te = split_triples(
            train_triples, args.train_frac, args.valid_frac, rng
        )
    else:
        tr = train_triples
        va = load_train2id_triples(valid_path)
        te = load_train2id_triples(test_path)

    os.makedirs(out_dir, exist_ok=True)
    write_triple_file(os.path.join(out_dir, "train.txt"), tr, id2ent, id2rel)
    write_triple_file(os.path.join(out_dir, "valid.txt"), va, id2ent, id2rel)
    write_triple_file(os.path.join(out_dir, "test.txt"), te, id2ent, id2rel)

    all_for_graph = tr + va + te
    true_str = build_true_string_set(all_for_graph, id2ent, id2rel)
    n_err = max(1, int(len(tr) * args.noise_ratio))
    corrupted = sample_corrupted_triples(tr, id2ent, id2rel, true_str, n_err, rng)
    err_path = os.path.join(out_dir, "errors", "%s-error.txt" % args.dataset_tag)
    write_errors(err_path, corrupted)

    rel_names_sorted = [id2rel[i] for i in sorted(id2rel.keys())]
    rules_path = os.path.join(
        out_dir,
        "rules",
        "%s-rules-top-%d.pkl" % (args.dataset_tag, args.rule_top_k),
    )
    write_empty_rules_pkl(rules_path, rel_names_sorted)

    print("✅ Wrote GOLD layout to:", out_dir)
    print("   train/valid/test lines:", len(tr), len(va), len(te))
    print("   synthetic errors:", len(corrupted), "->", err_path)
    print("   empty rules pkl:", rules_path)
    print("\nNext: cd GOLD-main && python gold.py --dataset %s --dataset_path %s ..." % (args.dataset_tag, out_dir))


if __name__ == "__main__":
    main()
