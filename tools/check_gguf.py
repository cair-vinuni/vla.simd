#!/usr/bin/env python3
"""
Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

Check a vla.cpp GGUF against the directory vla.simd's own converter writes
from the same upstream checkpoint:

  python tools/check_gguf.py <model.gguf | dir> <converted-dir> [--vla-simd-gguf PATH]

The GGUF is expanded with vla-simd-gguf (the engine's loader path) into a
temporary directory, then every file is compared: .meta/.txt as key/value
pairs, numerically; .bin element by element. A GGUF from an unchanged
checkpoint should come out IDENTICAL; anything else is listed with its size
and largest difference. Needs only numpy.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BF16 = {"emb.bin"}                          # stored as bf16 bits


def pairs(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        k, _, v = line.rstrip("\n").partition(" " if not path.endswith("text_pad.txt") else "\t")
        out[k] = v
    return out


def same_value(a, b):
    if a == b:
        return True
    try:
        return [float(x) for x in a.split()] == [float(x) for x in b.split()]
    except (AttributeError, ValueError):
        return False


def compare(got, ref):
    bad = 0
    for rel in sorted(set(walk(got)) | set(walk(ref))):
        a, b = os.path.join(got, rel), os.path.join(ref, rel)
        if not os.path.exists(a) or not os.path.exists(b):
            print(f"{rel:24s} only in {'the GGUF' if os.path.exists(a) else 'the converted dir'}")
            continue
        if rel.endswith(".bin"):
            dt = np.uint16 if os.path.basename(rel) in BF16 else np.uint32
            x, y = np.fromfile(a, dt), np.fromfile(b, dt)
            if x.size != y.size:
                print(f"{rel:24s} SIZE {x.size} vs {y.size} elements")
                bad += 1
                continue
            n = int(np.count_nonzero(x != y))
            if n == 0:
                print(f"{rel:24s} IDENTICAL ({x.size} elements)")
                continue
            if dt == np.uint16:
                x, y = (x.astype(np.uint32) << 16), (y.astype(np.uint32) << 16)
            d = np.abs(x.view(np.float32).astype(np.float64) - y.view(np.float32))
            print(f"{rel:24s} {n} of {x.size} differ, max |d| {d.max():.3e}")
            bad += 1
        elif rel.endswith((".meta", ".txt")):
            if os.path.basename(rel) in ("config.txt", "vocab.txt", "merges.txt"):
                ok = rel.endswith("config.txt") or open(a, "rb").read() == open(b, "rb").read()
                print(f"{rel:24s} {'same' if ok else 'DIFFERS'}")
                bad += not ok
                continue
            pa, pb = pairs(a), pairs(b)
            diff = {k: (pa.get(k), pb.get(k)) for k in sorted(set(pa) | set(pb))
                    if not same_value(pa.get(k), pb.get(k))}
            print(f"{rel:24s} {'same' if not diff else diff}")
            bad += bool(diff)
    return bad


def walk(root):
    for d, _, files in os.walk(root):
        for f in files:
            yield os.path.relpath(os.path.join(d, f), root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gguf", help="vla.cpp .gguf, or a directory holding one and its sidecars")
    p.add_argument("ref", help="directory written by tools/convert_*.py from the same checkpoint")
    p.add_argument("--vla-simd-gguf", default=os.path.join(HERE, "build", "vla-simd-gguf"))
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([args.vla_simd_gguf, "extract", args.gguf, tmp], stdout=subprocess.DEVNULL)
        if r.returncode:
            sys.exit(f"vla-simd-gguf extract failed ({r.returncode})")
        bad = compare(tmp, args.ref)
    print("ok" if not bad else f"{bad} file(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
