"""
Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

vla.cpp GGUF checkpoints for vla-simd-serve.

The engine loads a GGUF directly (src/io/files.h) and reads anything the GGUF
does not carry from files beside it. stage() builds that directory: a link to
the GGUF plus the sidecars, generated once and cached, so a read-only location
such as a Hub snapshot works too.

  smolvla   tok/vocab.txt, tok/merges.txt   SmolVLM2 tokenizer
            config.txt                      pad_token_id, chunk, num_steps
  turbovla  vocab.txt                       BERT WordPiece vocabulary
            stats.bin                       proprio mean/std, action min/max
            config.txt                      camera order
  octo      config.txt                      window, steps (everything else is in the GGUF)

A sidecar already beside the GGUF wins over a generated one, so a checkpoint
with its own statistics or tokenizer just ships them next to it.
"""

import hashlib
import json
import os
import struct

import numpy as np

SIDECARS = ("config.txt", "tok", "vocab.txt", "stats.bin")

SMOLVLA_TOKENIZER = "HuggingFaceTB/SmolVLM2-500M-Instruct"
BERT_TOKENIZER = "google-bert/bert-base-uncased"
TURBOVLA_STATS = ("H-EmbodVis/TurboVLA", "libero_all4_stats.json", "libero_all4_no_noops")
TURBOVLA_CAMS = ("agentview", "wrist")


def find_gguf(path):
    """The GGUF a model path names, or None: a .gguf file, or a directory with
    exactly one .gguf and no .meta files (the engine applies the same rule)."""
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path if path.endswith(".gguf") else None
    if not os.path.isdir(path):
        return None
    names = os.listdir(path)
    ggufs = [n for n in names if n.endswith(".gguf")]
    if len(ggufs) != 1 or any(n.endswith(".meta") for n in names):
        return None
    return os.path.join(path, ggufs[0])


def read_metadata(path):
    """GGUF key/values (strings and numbers; byte arrays are skipped)."""
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
              10: "<Q", 11: "<q", 12: "<d"}
    with open(path, "rb") as f:
        def read(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(n))[0]

        def string():
            return f.read(read("<Q")).decode("utf-8", "replace")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = read("<I"), read("<Q")
                if et in (0, 1):                 # u8/i8 blob, e.g. a sentencepiece model
                    f.seek(n, 1)
                    return None
                return [value(et) for _ in range(n)]
            return read(scalar[t])

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = read("<I")
        if version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {version} is not supported")
        read("<Q")                               # tensor count
        meta = {}
        for _ in range(read("<Q")):
            key = string()
            meta[key] = value(read("<I"))
    return meta


def _hub_file(repo, name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name)


def _smolvla_tokenizer(out_dir, repo):
    """Flat vocab/merges for the byte-level BPE tokenizer, as
    tools/convert_hf_safetensors.py writes them."""
    with open(_hub_file(repo, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for tok, i in vocab.items():
            f.write(f"{i}\t{tok}\n")
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for m in merges:
            f.write((m if isinstance(m, str) else f"{m[0]} {m[1]}") + "\n")
    pad = next((a["id"] for a in tj.get("added_tokens", []) if a["content"] == "<|im_end|>"),
               vocab.get("<|im_end|>", 2))
    return pad


def _turbovla_stats(path, repo, name, key, meta):
    with open(_hub_file(repo, name)) as f:
        st = json.load(f)[key]
    s, a = meta["turbovla.state_dim"], meta["turbovla.action_dim"]
    parts = [(st["proprio"]["mean"], s), (st["proprio"]["std"], s),
             (st["action"]["min"], a), (st["action"]["max"], a)]
    for v, n in parts:
        if len(v) != n:
            raise ValueError(f"{repo}/{name}[{key}]: statistics are {len(v)}-wide, the model {n}")
    np.concatenate([np.asarray(v, np.float32) for v, _ in parts]).tofile(path)


def stage(path, model, cache_root=None):
    """Directory the engine loads `path` (a GGUF, or a directory holding one)
    from, with every sidecar the architecture needs. Returns `path` unchanged
    when it is not a GGUF."""
    gguf = find_gguf(path)
    if gguf is None:
        return path
    meta = read_metadata(gguf)
    arch = meta.get("general.architecture")
    if arch != model:
        raise SystemExit(f"{gguf} is a vla.cpp '{arch}' GGUF, not {model} (pass --model {arch})")

    real = os.path.realpath(gguf)
    cache_root = os.path.expanduser(
        cache_root or os.environ.get("VLA_SIMD_CACHE", "~/.cache/vla_simd/gguf"))
    stem = os.path.splitext(os.path.basename(real))[0]
    out = os.path.join(cache_root, f"{stem}-{hashlib.sha1(real.encode()).hexdigest()[:10]}")
    os.makedirs(out, exist_ok=True)
    link = os.path.join(out, os.path.basename(real))
    if os.path.islink(link) and os.readlink(link) != real:
        os.unlink(link)
    if not os.path.lexists(link):
        os.symlink(real, link)

    # sidecars shipped beside the GGUF win: link them in instead of generating
    src_dir = os.path.dirname(gguf)
    for name in SIDECARS:
        src, dst = os.path.join(src_dir, name), os.path.join(out, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.realpath(src), dst)

    def missing(name):
        return not os.path.exists(os.path.join(out, name))

    config = []
    if arch == "smolvla":
        if missing("tok"):
            pad = _smolvla_tokenizer(os.path.join(out, "tok"), SMOLVLA_TOKENIZER)
            config.append(f"pad_token_id {pad}")
        config += [f"chunk {meta['smolvla.chunk_size']}", f"num_steps {meta['smolvla.num_steps']}"]
    elif arch == "turbovla":
        if missing("vocab.txt"):
            os.symlink(_hub_file(BERT_TOKENIZER, "vocab.txt"), os.path.join(out, "vocab.txt"))
        if missing("stats.bin"):
            _turbovla_stats(os.path.join(out, "stats.bin"), *TURBOVLA_STATS, meta)
        config += [f"cam{i} {c}" for i, c in enumerate(TURBOVLA_CAMS)]
    elif arch == "octo":
        config += [f"window {meta['octo.window_size']}", f"steps {meta['octo.diffusion.steps']}"]
    else:
        raise SystemExit(f"{gguf}: vla.cpp architecture '{arch}' is not supported "
                         "(smolvla, turbovla and octo are)")
    if missing("config.txt") and config:
        with open(os.path.join(out, "config.txt"), "w") as f:
            f.write("\n".join(config) + "\n")
    return out
