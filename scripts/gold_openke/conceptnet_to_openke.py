#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export GOLD ConceptNet text files -> OpenKE benchmark layout (entity2id, relation2id, train2id, ...).

Phải khớp cách GOLD gán id trong data_loader.Dataset.read() cho conceptnet:
  mỗi dòng: relation \\t head \\t tail
  thứ tự gán id: entity[head], relation[relation], entity[tail] (giống GOLD).

Đầu ra dùng cho gold_scores_to_openke_train.py (chuỗi h/r/t trùng TSV của gold.py).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple


class _OrderedMap:
    """Giống MyDict trong GOLD: id tăng khi gặp chuỗi mới."""

    def __init__(self) -> None:
        self.k2v: Dict[str, int] = {}
        self.v2k: Dict[int, str] = {}
        self.cnt = 0

    def __getitem__(self, key: str) -> int:
        if key not in self.k2v:
            self.k2v[key] = self.cnt
            self.v2k[self.cnt] = key
            self.cnt += 1
        return self.k2v[key]


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _parse_conceptnet_line(line: str) -> Tuple[str, str, str]:
    sep = line.strip().split("\t")
    if len(sep) != 3:
        raise ValueError("Bad line (need 3 cols r\\th\\tt): %r" % line)
    r_s, h_s, t_s = sep[0].strip(), sep[1].strip(), sep[2].strip()
    return h_s, r_s, t_s


def _assign_ids_like_gold(lines: List[str], entity: _OrderedMap, relation: _OrderedMap) -> None:
    """Một lượt giống GOLD: đọc train+valid+test nối tiếp."""
    for line in lines:
        if not line.strip():
            continue
        h_s, r_s, t_s = _parse_conceptnet_line(line)
        entity[h_s]
        relation[r_s]
        entity[t_s]


def _lines_to_triple_ids(
    lines: List[str], entity: _OrderedMap, relation: _OrderedMap
) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    seen = set()
    for line in lines:
        if not line.strip():
            continue
        h_s, r_s, t_s = _parse_conceptnet_line(line)
        h, r, t = entity[h_s], relation[r_s], entity[t_s]
        key = (h, r, t)
        if key in seen:
            continue
        seen.add(key)
        out.append((h, r, t))
    return out


def _write_map(path: str, m: _OrderedMap) -> None:
    """OpenKE: dòng 1 = số lượng; các dòng sau name\\tid."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("%d\n" % m.cnt)
        for i in range(m.cnt):
            name = m.v2k[i]
            f.write("%s\t%d\n" % (name, i))


def _write_train2id(path: str, triples: List[Tuple[int, int, int]]) -> None:
    """OpenKE: dòng 1 = số triple; mỗi dòng head_id tail_id rel_id."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("%d\n" % len(triples))
        for h, r, t in triples:
            f.write("%d %d %d\n" % (h, t, r))


def main() -> None:
    ap = argparse.ArgumentParser(description="GOLD ConceptNet -> OpenKE folder")
    ap.add_argument(
        "--conceptnet_dir",
        required=True,
        help="Thư mục dataset/conceptnet (train.txt, valid.txt, test.txt)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Thư mục ghi entity2id.txt, relation2id.txt, train2id.txt, ...",
    )
    args = ap.parse_args()

    cn = os.path.abspath(args.conceptnet_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_p = os.path.join(cn, "train.txt")
    valid_p = os.path.join(cn, "valid.txt")
    test_p = os.path.join(cn, "test.txt")
    for p in (train_p, valid_p, test_p):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    entity = _OrderedMap()
    relation = _OrderedMap()

    all_lines = []
    for p in (train_p, valid_p, test_p):
        all_lines.extend(_read_lines(p))

    _assign_ids_like_gold(all_lines, entity, relation)

    train_lines = _read_lines(train_p)
    valid_lines = _read_lines(valid_p)
    test_lines = _read_lines(test_p)

    train_triples = _lines_to_triple_ids(train_lines, entity, relation)
    valid_triples = _lines_to_triple_ids(valid_lines, entity, relation)
    test_triples = _lines_to_triple_ids(test_lines, entity, relation)

    _write_map(os.path.join(out_dir, "entity2id.txt"), entity)
    _write_map(os.path.join(out_dir, "relation2id.txt"), relation)
    _write_train2id(os.path.join(out_dir, "train2id.txt"), train_triples)
    _write_train2id(os.path.join(out_dir, "valid2id.txt"), valid_triples)
    _write_train2id(os.path.join(out_dir, "test2id.txt"), test_triples)

    print("Wrote OpenKE layout ->", out_dir)
    print("  entities:", entity.cnt, "relations:", relation.cnt)
    print("  train/valid/test triples:", len(train_triples), len(valid_triples), len(test_triples))


if __name__ == "__main__":
    main()
