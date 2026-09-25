"""
Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

Export the t5-base sentencepiece unigram tokenizer for the C++ engine. The Octo
converters call export() to put it inside their GGUF; the command line writes
the bare vocab.txt.

Run from the repo root with the reference venv (setup: see convert_octo.py;
only transformers + sentencepiece are actually needed here):
  python tools/convert_t5_tokenizer.py [OUT_DIR] [--ckpt ID_OR_DIR]

Outputs (build/octo/tok/):
  vocab.txt    one line per id: piece<TAB>score  (32100: 32000 sp + 100 extra_ids)
"""
import argparse
import os


def export(out, ckpt="t5-base"):
    """vocab.txt for the engine's T5 unigram tokenizer, into directory `out`.
    Pieces and scores come from spiece.model itself, so this does not depend on
    which transformers release still exposes the slow tokenizer's sp_model."""
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    os.makedirs(out, exist_ok=True)
    local = os.path.join(os.path.expanduser(ckpt), "spiece.model")
    sp = sentencepiece.SentencePieceProcessor(
        model_file=local if os.path.isfile(local) else hf_hub_download(ckpt, "spiece.model"))
    fast = AutoTokenizer.from_pretrained(ckpt)
    with open(f"{out}/vocab.txt", "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
        for i in range(sp.get_piece_size(), len(fast)):
            f.write(f"{fast.convert_ids_to_tokens(i)}\t0.0\n")


def export_encoder(out, ckpt="google-t5/t5-base", n_tokens=16):
    """t5.meta / t5.bin in convert_octo.py's layout, from the Hugging Face
    weights. Octo keeps T5-base frozen, and these are bit-identical to the copy
    inside the Octo checkpoint, so a finetune that ships without its language
    tower gets it from here."""
    import numpy as np
    import torch
    from transformers import T5EncoderModel

    m = T5EncoderModel.from_pretrained(ckpt, torch_dtype=torch.float32)
    c, enc = m.config, m.encoder

    def f32(t):
        return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)

    rel = enc.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    with open(f"{out}/t5.meta", "w") as f:
        f.write(f"d_model {c.d_model}\nn_layers {c.num_layers}\nn_heads {c.num_heads}\n"
                f"d_kv {c.d_kv}\nd_ff {c.d_ff}\n")
        f.write(f"vocab {m.shared.weight.shape[0]}\n")
        f.write(f"n_buckets {rel.shape[0]}\nmax_dist {c.relative_attention_max_distance}\n"
                f"eps {c.layer_norm_epsilon:g}\nn_tokens {n_tokens}\n")
    with open(f"{out}/t5.bin", "wb") as f:
        f32(m.shared.weight).tofile(f)
        f32(rel).tofile(f)
        for blk in enc.block:
            att, ff = blk.layer[0], blk.layer[1]
            f32(att.layer_norm.weight).tofile(f)
            for proj in (att.SelfAttention.q, att.SelfAttention.k, att.SelfAttention.v,
                         att.SelfAttention.o):
                f32(proj.weight).tofile(f)
            f32(ff.layer_norm.weight).tofile(f)
            f32(ff.DenseReluDense.wi.weight).tofile(f)
            f32(ff.DenseReluDense.wo.weight).tofile(f)
        f32(enc.final_layer_norm.weight).tofile(f)


if __name__ == "__main__":
    _p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _p.add_argument("out", nargs="?", default="build/octo/tok",
                    help="output dir (default: build/octo/tok)")
    _p.add_argument("--ckpt", default="t5-base", help="tokenizer to export (default: t5-base)")
    _a = _p.parse_args()
    export(_a.out, _a.ckpt)
    print(f"done -> {_a.out}")
