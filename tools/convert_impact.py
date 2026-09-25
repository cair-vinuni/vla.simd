#!/usr/bin/env python3
"""
Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

Convert a lerobot IMPACT policy to the vla.simd GGUF the engine loads.

    # from a trained checkpoint
    python tools/convert_impact.py \
      --ckpt ~/work/lerobot/outputs/train/impact_so101_tape/checkpoints/last/pretrained_model \
      --out build/impact_so101/impact_so101.gguf

    # random weights, for benchmarking before training finishes
    python tools/convert_impact.py --random --out build/impact_rand.gguf

Needs the lerobot venv (torch, torchvision, transformers) and the IMPACT policy on
the path; see lerobot/src/lerobot/policies/impact/.

Packed into one GGUF (tools/_gguf.py), as these files:
  config.txt      image size, camera order, T5 special ids
  vision.meta/bin ResNet-18 + FiLM points, every BatchNorm folded into the conv
  text.meta/bin   T5-small encoder (nn::T5Encoder layout) + text proj + text pos + FiLM head
  impact.meta/bin the DETR transformer: projections, 1D pos, encoder, decoder, head
  stats.bin       state/action mean+std and the per-camera image mean+std
  vocab.txt       T5 sentencepiece vocabulary, piece<TAB>score per line
  vocab_map.bin   int32[vocab_full]: full id -> engine id (identity unless pruned)

A note on why the random mode exists at all: IMPACT is a new architecture, so there
is no upstream checkpoint to be faithful to. Random weights at a fixed seed still
establish that the C++ forward pass computes the same function as the torch module,
which is the only thing parity can ever establish. They say nothing about the policy.
"""

import argparse
import os
import sys

import numpy as np
import torch

from _common import Arena, dump_detr, dump_resnet18, log, to_numpy, write_meta
from _gguf import gguf_output


# ---------------------------------------------------------------------------
# vision: ResNet-18 + FiLM points
# ---------------------------------------------------------------------------
def dump_vision(sd, policy, out_dir):
    """ResNet-18 truncated at layer4, every BN folded away, FiLM points recorded.

    IMPACT's FiLMResNet runs the stages explicitly (ACT's IntermediateLayerGetter
    cannot express a hook between them), so the state dict names them
    `model.backbone.stages.{0..3}.{0,1}` rather than `model.backbone.layer{1..4}.{0,1}`.
    The tensors are the same torchvision ones either way.
    """
    backbone = getattr(getattr(policy, "config", None), "vision_backbone", None)
    if backbone not in (None, "resnet18"):
        sys.exit(f"vision_backbone is {backbone!r}; this exporter only handles resnet18 "
                 "(2 BasicBlocks per stage). Extend the loop below before using it.")

    arena = Arena()
    meta = dump_resnet18(arena, sd, "model.backbone.conv1", "model.backbone.bn1",
                         [f"model.backbone.stages.{i}" for i in range(4)])
    blocks = len(meta) - 8
    # FiLM is applied at the END of each stage, after the residual add and its
    # ReLU -- so it follows the stage's LAST block.
    film_after = list(range(1, blocks, 2))

    use_film = getattr(policy.config, "use_film", True)
    if use_film:
        meta += [f"film_after {i}" for i in film_after]

    write_meta(out_dir, "vision", meta)
    n = arena.write(os.path.join(out_dir, "vision.bin"))
    log(f"  vision.bin  {n * 4 / 1e6:8.1f} MB   resnet18, {blocks} blocks, "
        f"FiLM after {film_after if use_film else 'none'}")
    return film_after if use_film else []


# ---------------------------------------------------------------------------
# text: T5-small encoder + the three heads
# ---------------------------------------------------------------------------
def dump_text(policy, out_dir, film_channels):
    """The frozen T5-small encoder in nn::T5Encoder's layout, then, appended past
    it in the same arena and read from the loader's `tail`: the text projection,
    the learned text position table, and the FiLM head.

    The order here is the format. `ImpactText::load` walks it in exactly this
    sequence and checks the total, so an extra or missing tensor is a load error
    rather than a silently shifted arena.
    """
    model = policy.model
    t5 = model.text.model
    cfg = t5.config
    D, FF = cfg.d_model, cfg.d_ff
    vocab = cfg.vocab_size
    assert cfg.feed_forward_proj == "relu", f"T5 FFN is {cfg.feed_forward_proj}, not relu"
    assert not cfg.is_gated_act, "gated T5 (v1.1) needs a second wi; this loader has one"

    arena = Arena()
    arena.add(t5.encoder.embed_tokens.weight, (vocab, D))
    # The relative position bias lives on layer 0 only and is shared by every
    # layer, which is why the engine keeps one [n_buckets, n_heads] table.
    rel = t5.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    arena.add(rel, (cfg.relative_attention_num_buckets, cfg.num_heads))

    for blk in t5.encoder.block:
        attn = blk.layer[0]
        arena.add(attn.layer_norm.weight, (D,))
        inner = cfg.num_heads * cfg.d_kv
        arena.add(attn.SelfAttention.q.weight, (inner, D))
        arena.add(attn.SelfAttention.k.weight, (inner, D))
        arena.add(attn.SelfAttention.v.weight, (inner, D))
        arena.add(attn.SelfAttention.o.weight, (D, inner))
        ff = blk.layer[1]
        arena.add(ff.layer_norm.weight, (D,))
        arena.add(ff.DenseReluDense.wi.weight, (FF, D))
        arena.add(ff.DenseReluDense.wo.weight, (D, FF))
    arena.add(t5.encoder.final_layer_norm.weight, (D,))
    encoder_floats = arena.n

    dim = policy.config.dim_model
    n_text = policy.config.tokenizer_max_length

    arena.add(model.text_input_proj.weight, (dim, D)).add(model.text_input_proj.bias, (dim,))
    arena.add(model.text_pos_embed.weight, (n_text, dim))

    film_total = int(sum(film_channels)) if film_channels else 0
    if film_channels:
        head = model.film_head.net
        assert isinstance(head, torch.nn.Linear), (
            "film_hidden_dim > 0 builds an MLP head; the engine's arena holds one "
            "linear. Export the hidden layer too, or train with film_hidden_dim=0.")
        arena.add(head.weight, (2 * film_total, D)).add(head.bias, (2 * film_total,))

    write_meta(out_dir, "text", [
        f"d_model {D}",
        f"n_layers {cfg.num_layers}",
        f"n_heads {cfg.num_heads}",
        f"d_kv {cfg.d_kv}",
        f"d_ff {FF}",
        f"vocab {vocab}",
        f"n_buckets {cfg.relative_attention_num_buckets}",
        f"max_dist {cfg.relative_attention_max_distance}",
        f"n_tokens {n_text}",
        f"eps {float(cfg.layer_norm_epsilon):g}",
        f"proj_dim {dim}",
        f"n_text {n_text}",
        f"n_film {len(film_channels)}",
        f"film_total {film_total}",
        f"encoder_floats {encoder_floats}",
    ])
    n = arena.write(os.path.join(out_dir, "text.bin"))
    log(f"  text.bin    {n * 4 / 1e6:8.1f} MB   T5-small encoder (vocab {vocab}) "
        f"+ proj + pos + FiLM({film_total})")
    return vocab


# ---------------------------------------------------------------------------
# transformer
# ---------------------------------------------------------------------------
def dump_transformer(sd, policy, out_dir):
    cfg = policy.config
    arena = Arena()
    meta = dump_detr(arena, sd, cfg, cfg.robot_state_feature.shape[0], cfg.action_feature.shape[0])
    write_meta(out_dir, "impact", meta + [f"n_text {cfg.tokenizer_max_length}", "ln_eps 1e-5"])
    n = arena.write(os.path.join(out_dir, "impact.bin"))
    log(f"  impact.bin  {n * 4 / 1e6:8.1f} MB   {cfg.n_encoder_layers} enc + "
        f"{cfg.n_decoder_layers} dec, chunk {cfg.chunk_size}")


# ---------------------------------------------------------------------------
# stats, config, tokenizer
# ---------------------------------------------------------------------------
def dump_stats(stats, cam_keys, out_dir, state_dim, action_dim):
    """state mean/std, action mean/std, then per-camera image mean/std."""
    def get(key, field, n, default):
        d = stats.get(key) or {}
        v = d.get(field)
        return np.full(n, default, np.float32) if v is None else \
            np.asarray(to_numpy(v), np.float32).reshape(-1)[:n]

    parts = [
        get("observation.state", "mean", state_dim, 0.0),
        get("observation.state", "std", state_dim, 1.0),
        get("action", "mean", action_dim, 0.0),
        get("action", "std", action_dim, 1.0),
    ]
    for k in cam_keys:
        parts.append(get(k, "mean", 3, 0.0))
        parts.append(get(k, "std", 3, 1.0))
    np.concatenate(parts).astype(np.float32).tofile(os.path.join(out_dir, "stats.bin"))


def dump_config(out_dir, img_h, img_w, cam_names, vocab_full, unk_id, instruction=None):
    lines = [
        f"img_h {img_h}",
        f"img_w {img_w}",
        f"n_cams {len(cam_names)}",
        "norm_eps 1e-8",
        f"vocab_full {vocab_full}",
        f"unk_id {unk_id}",
    ]
    lines += [f"cam{i} {n}" for i, n in enumerate(cam_names)]
    # The policy server reads this as the default task, so a rollout works without
    # --task. Free text after the first space; read_config splits on that.
    if instruction:
        lines.append(f"instruction {instruction}")
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def dump_tokenizer(out_dir, vocab_full):
    """The T5 sentencepiece vocabulary, plus an identity vocab_map.

    MicroVLA prunes this table to its instruction corpus and carries the mapping.
    IMPACT keeps the full vocabulary: the tower runs once per episode, so the
    pruning would buy memory (66 MB of embedding) and no latency, and it would
    close the vocabulary on a policy whose whole point is being retargetable by
    sentence. The map is written anyway, as the identity, so the engine's loader
    and a future pruned checkpoint share one code path.
    """
    import json

    from transformers import AutoTokenizer

    fast = AutoTokenizer.from_pretrained("google-t5/t5-small")

    # The unigram pieces and their scores come from the fast tokenizer's own
    # serialized model. transformers 5.x dropped `sp_model` off the slow T5
    # tokenizer, and the backend JSON is the same table sentencepiece would have
    # handed over -- ordered by id, which is what vocab.txt's line numbering means.
    spec = json.loads(fast.backend_tokenizer.to_str())
    pieces = spec.get("model", {}).get("vocab") or []
    if not pieces:
        sys.exit("could not read the unigram vocabulary from the T5 tokenizer; "
                 "tools/convert_t5_tokenizer.py has the sentencepiece path")

    table = [None] * vocab_full
    for i, entry in enumerate(pieces):
        if i >= vocab_full:
            break
        table[i] = (entry[0], float(entry[1]))
    for i in range(len(pieces), vocab_full):
        table[i] = (fast.convert_ids_to_tokens(i) or "", 0.0)

    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for piece, score in table:
            f.write(f"{piece}\t{score}\n")

    np.arange(vocab_full, dtype=np.int32).tofile(os.path.join(out_dir, "vocab_map.bin"))
    return fast


def load_dataset_stats(ckpt_dir):
    """`{feature: {"mean":…, "std":…}}` from the checkpoint's normalizer, or {}.

    A trained checkpoint carries the dataset statistics in its preprocessor rather
    than in `model.safetensors`, so a converter that only reads the model emits
    identity normalization and the engine silently mis-scales both the input state
    and the output actions. That is invisible to a parity run whose reference is fed
    the same un-normalized inputs, which is exactly how it survived until now.

    A random-init policy has no statistics and legitimately gets identity.
    """
    import glob
    if not ckpt_dir:
        return {}
    from safetensors import safe_open
    stats, want = {}, ("mean", "std")
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*normalizer*.safetensors"))):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                feat, _, field = k.rpartition(".")
                if field in want:
                    stats.setdefault(feat, {})[field] = to_numpy(sf.get_tensor(k))
        if stats:
            break
    return stats


# ---------------------------------------------------------------------------
def build_random_policy(img_h, img_w, cam_keys, state_dim, action_dim, seed, **overrides):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.impact.configuration_impact import IMPACTConfig
    from lerobot.policies.impact.modeling_impact import IMPACTPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    torch.manual_seed(seed)
    cfg = IMPACTConfig(
        input_features={
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)) for k in cam_keys},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        pretrained_backbone_weights=None,
        device="cpu",
        **overrides,
    )
    return IMPACTPolicy(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="lerobot pretrained_model directory")
    ap.add_argument("--random", action="store_true", help="random weights at --seed")
    ap.add_argument("--out", default="build/impact",
                    help="output .gguf, or a dir to write <dir>/<dir>.gguf in")
    ap.add_argument("--img", default="480x640", help="HxW of the camera frames")
    ap.add_argument("--cams", default="observation.images.front,observation.images.wrist")
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--instruction", default="Put the tape into the box")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.ckpt and not args.random:
        ap.error("pass --ckpt DIR or --random")

    img_h, img_w = (int(x) for x in args.img.lower().split("x"))
    cam_keys = args.cams.split(",")

    source = args.ckpt or f"random:{args.seed}"
    if args.ckpt:
        if not os.path.isdir(args.ckpt):
            from huggingface_hub import snapshot_download
            args.ckpt = snapshot_download(args.ckpt)
        from lerobot.policies.impact.modeling_impact import IMPACTPolicy
        log(f"loading {args.ckpt}")
        policy = IMPACTPolicy.from_pretrained(args.ckpt)
        # A checkpoint trained on a GPU carries device="cuda" in its config and
        # lands there. Everything below is CPU arithmetic, so pull the whole
        # policy over rather than half-moving tensors.
        policy.to("cpu")
        policy.config.device = "cpu"
        cam_keys = [k for k in policy.config.image_features]
        shape = policy.config.image_features[cam_keys[0]].shape
        img_h, img_w = shape[1], shape[2]
    else:
        log(f"random weights, seed {args.seed}")
        policy = build_random_policy(img_h, img_w, cam_keys, args.state_dim,
                                     args.action_dim, args.seed)

    policy.eval()
    sd = policy.state_dict()
    cfg = policy.config
    if getattr(cfg, "pre_norm", False) or "model.encoder.norm.weight" in sd:
        sys.exit("pre_norm checkpoints are not supported by the engine (post-norm only)")
    if getattr(cfg, "feedforward_activation", "relu") != "relu":
        sys.exit(f"only relu feedforward is supported, got {cfg.feedforward_activation}")
    if sd["model.encoder_1d_feature_pos_embed.weight"].shape[0] != 2:
        sys.exit("the engine builds exactly [latent, state] 1-D tokens; env_state is not supported")
    log(f"converting -> {args.out}  ({img_h}x{img_w}, cams {cam_keys})")

    with gguf_output(args.out, "impact", source=source) as out:
        film_after = dump_vision(sd, policy, out.dir)
        film_channels = policy.model.backbone.stage_channels if film_after else []
        vocab_full = dump_text(policy, out.dir, film_channels)
        dump_transformer(sd, policy, out.dir)

        state_dim = policy.config.robot_state_feature.shape[0]
        action_dim = policy.config.action_feature.shape[0]
        stats = load_dataset_stats(args.ckpt)
        if args.ckpt and not {"observation.state", "action"} <= stats.keys():
            sys.exit(f"no state/action mean+std in *normalizer*.safetensors under {args.ckpt}")
        if stats:
            log(f"  stats           dataset statistics from the checkpoint's normalizer "
                f"({len(stats)} features)")
        else:
            log("  stats           IDENTITY - no normalizer found (random init?)")
        dump_stats(stats, cam_keys, out.dir, state_dim, action_dim)
        tok = dump_tokenizer(out.dir, vocab_full)
        out.layout["vocab_map.bin"] = [("I32", None)]
        dump_config(out.dir, img_h, img_w, [k.split(".")[-1] for k in cam_keys],
                    vocab_full, tok.unk_token_id if tok.unk_token_id is not None else 2,
                    args.instruction)


if __name__ == "__main__":
    main()
