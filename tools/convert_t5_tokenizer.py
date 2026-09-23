"""
Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

Export the t5-base sentencepiece unigram tokenizer for the C++ engine.

Run from the repo root with the reference venv (setup: see convert_octo.py;
only transformers + sentencepiece are actually needed here):
  python tools/convert_t5_tokenizer.py [OUT_DIR] [--ckpt ID_OR_DIR]

Outputs (build/octo/tok/):
  vocab.txt    one line per id: piece<TAB>score  (32100: 32000 sp + 100 extra_ids)
"""
import argparse
import os

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("out", nargs="?", default="build/octo/tok",
                help="output dir (default: build/octo/tok)")
_p.add_argument("--ckpt", default="t5-base", help="tokenizer to export (default: t5-base)")
_a = _p.parse_args()

OUT = _a.out
os.makedirs(OUT, exist_ok=True)

from transformers import AutoTokenizer, T5Tokenizer

slow = T5Tokenizer.from_pretrained(_a.ckpt, legacy=True)
sp = slow.sp_model
fast = AutoTokenizer.from_pretrained(_a.ckpt)

with open(f"{OUT}/vocab.txt", "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
    for i in range(sp.get_piece_size(), fast.vocab_size):
        f.write(f"{fast.convert_ids_to_tokens(i)}\t0.0\n")

print(f"done -> {OUT}")
