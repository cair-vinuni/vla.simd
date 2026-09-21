#!/usr/bin/env python3
"""
Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

Convert a lerobot IMPACT policy to the C++ engine's arenas.

    # from a trained checkpoint
    python tools/impact/convert_impact.py \
      --checkpoint ~/work/lerobot/outputs/train/impact_so101_tape/checkpoints/last/pretrained_model \
      --out build/impact_so101

    # random weights, for benchmarking before training finishes
    python tools/impact/convert_impact.py --random --out build/impact_rand

Needs the lerobot venv (torch, torchvision, transformers) and the IMPACT policy on
the path; see lerobot/src/lerobot/policies/impact/.

Outputs (out_dir/):
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# weight arena helpers -- the cursor these mirror is `take()` in the C++ loaders
# ---------------------------------------------------------------------------
class Arena:
    """Append-only float32 blob."""

    def __init__(self):
        self.parts = []
        self.n = 0

    def add(self, t, shape=None):
        a = np.ascontiguousarray(_to_numpy(t), dtype=np.float32)
        if shape is not None:
            assert tuple(a.shape) == tuple(shape), f"expected {shape}, got {tuple(a.shape)}"
        self.parts.append(a.reshape(-1))
        self.n += a.size
        return self

    def write(self, path):
        blob = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
        blob.tofile(path)
        return blob.size


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


def write_meta(out_dir, stem, lines):
    with open(os.path.join(out_dir, f"{stem}.meta"), "w") as f:
        f.write("\n".join(lines) + "\n")


def linear(arena, sd, prefix, n_out=None, n_in=None):
    """nn.Linear -> W [N, K] then bias [N], the layout nn::Linear::init expects."""
    w = sd[f"{prefix}.weight"]
    b = sd[f"{prefix}.bias"]
    if n_out is not None:
        assert tuple(w.shape) == (n_out, n_in), f"{prefix}: {tuple(w.shape)} != {(n_out, n_in)}"
    arena.add(w).add(b)


def layernorm(arena, sd, prefix):
    arena.add(sd[f"{prefix}.weight"]).add(sd[f"{prefix}.bias"])


def mha(arena, sd, prefix, dim):
    """nn.MultiheadAttention packs qkv in one in_proj_weight [3D, D]; the engine
    keeps four separate nn::Linear, so split it here once."""
    w = sd[f"{prefix}.in_proj_weight"]
    b = sd[f"{prefix}.in_proj_bias"]
    assert tuple(w.shape) == (3 * dim, dim), f"{prefix}: {tuple(w.shape)}"
    for i in range(3):  # q, k, v
        arena.add(w[i * dim:(i + 1) * dim]).add(b[i * dim:(i + 1) * dim])
    linear(arena, sd, f"{prefix}.out_proj", dim, dim)


def fold_bn(conv_w, bn_prefix, sd, eps=1e-5):
    """conv (no bias) followed by a frozen BatchNorm -> conv weights + a bias.

    scale = gamma / sqrt(running_var + eps); W' = W * scale, b' = beta - mean*scale.
    Exact rather than an approximation: ACT freezes every BN (FrozenBatchNorm2d), so
    the running statistics never move.
    """
    gamma = sd[f"{bn_prefix}.weight"].to(torch.float32)
    beta = sd[f"{bn_prefix}.bias"].to(torch.float32)
    mean = sd[f"{bn_prefix}.running_mean"].to(torch.float32)
    var = sd[f"{bn_prefix}.running_var"].to(torch.float32)

    scale = gamma / torch.sqrt(var + eps)
    w = conv_w.to(torch.float32) * scale.reshape(-1, 1, 1, 1)
    b = beta - mean * scale
    return w, b


def conv_nhwc(arena, w, b):
    """torch conv weight [Cout, Cin, k, k] -> the engine's [Cout, k, k, Cin]."""
    arena.add(w.permute(0, 2, 3, 1).contiguous()).add(b)


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
    meta = []
    arena = Arena()

    stem_w = sd["model.backbone.conv1.weight"]
    cout, cin, k, _ = stem_w.shape
    w, b = fold_bn(stem_w, "model.backbone.bn1", sd)
    conv_nhwc(arena, w, b)
    meta += [
        f"in_ch {cin}",
        f"stem_out {cout}",
        f"stem_k {k}",
        "stem_stride 2",
        f"stem_pad {k // 2}",
        "pool_k 3",
        "pool_stride 2",
        "pool_pad 1",
    ]

    backbone = getattr(getattr(policy, "config", None), "vision_backbone", None)
    if backbone not in (None, "resnet18"):
        sys.exit(f"vision_backbone is {backbone!r}; this exporter only handles resnet18 "
                 "(2 BasicBlocks per stage). Extend the loop below before using it.")

    film_after = []
    blocks = 0
    for stage in range(4):
        if f"model.backbone.stages.{stage}.2.conv1.weight" in sd:
            sys.exit(f"model.backbone.stages.{stage} has more than 2 blocks -- not a resnet18")
        for blk in range(2):
            p = f"model.backbone.stages.{stage}.{blk}"
            if f"{p}.conv1.weight" not in sd:
                sys.exit(f"missing {p}.conv1.weight -- is this a resnet18 backbone?")
            c1 = sd[f"{p}.conv1.weight"]
            bcout, bcin, bk, _ = c1.shape
            assert bk == 3, f"{p}: only basic blocks (3x3) are supported, got k={bk}"
            stride = 2 if (stage > 0 and blk == 0) else 1
            has_down = f"{p}.downsample.0.weight" in sd

            w, b = fold_bn(c1, f"{p}.bn1", sd)
            conv_nhwc(arena, w, b)
            w, b = fold_bn(sd[f"{p}.conv2.weight"], f"{p}.bn2", sd)
            conv_nhwc(arena, w, b)
            if has_down:
                w, b = fold_bn(sd[f"{p}.downsample.0.weight"], f"{p}.downsample.1", sd)
                conv_nhwc(arena, w, b)

            meta.append(f"block {bcin} {bcout} {stride} {int(has_down)}")
            blocks += 1
        # FiLM is applied at the END of each stage, after the residual add and its
        # ReLU -- so it follows the stage's LAST block.
        film_after.append(blocks - 1)

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
    else:
        # A model without FiLM still needs the field present so the two arenas can
        # be checked against each other; a zero-width head is the honest encoding.
        pass

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
    d = cfg.dim_model
    arena = Arena()

    # The camera-token projection is a 1x1 conv; over tokens that is a plain
    # linear, so it is stored [d, cin] like every other nn::Linear.
    w = sd["model.encoder_img_feat_input_proj.weight"]
    assert w.shape[2:] == (1, 1), f"img proj is {tuple(w.shape)}, expected a 1x1 conv"
    arena.add(w.reshape(w.shape[0], w.shape[1])).add(sd["model.encoder_img_feat_input_proj.bias"])

    state_dim = cfg.robot_state_feature.shape[0]
    linear(arena, sd, "model.encoder_robot_state_input_proj", d, state_dim)

    # At inference the CVAE latent is all zeros, so encoder_latent_input_proj(0)
    # collapses to its own bias. The engine stores that constant token and never
    # sees the style encoder at all.
    lat_w = sd["model.encoder_latent_input_proj.weight"]
    lat_b = sd["model.encoder_latent_input_proj.bias"]
    assert tuple(lat_w.shape) == (d, cfg.latent_dim)
    arena.add(lat_b, (d,))

    arena.add(sd["model.encoder_1d_feature_pos_embed.weight"])
    n_1d = sd["model.encoder_1d_feature_pos_embed.weight"].shape[0]

    for i in range(cfg.n_encoder_layers):
        p = f"model.encoder.layers.{i}"
        mha(arena, sd, f"{p}.self_attn", d)
        layernorm(arena, sd, f"{p}.norm1")
        linear(arena, sd, f"{p}.linear1", cfg.dim_feedforward, d)
        linear(arena, sd, f"{p}.linear2", d, cfg.dim_feedforward)
        layernorm(arena, sd, f"{p}.norm2")

    for i in range(cfg.n_decoder_layers):
        p = f"model.decoder.layers.{i}"
        mha(arena, sd, f"{p}.self_attn", d)
        layernorm(arena, sd, f"{p}.norm1")
        mha(arena, sd, f"{p}.multihead_attn", d)
        layernorm(arena, sd, f"{p}.norm2")
        linear(arena, sd, f"{p}.linear1", cfg.dim_feedforward, d)
        linear(arena, sd, f"{p}.linear2", d, cfg.dim_feedforward)
        layernorm(arena, sd, f"{p}.norm3")

    arena.add(sd["model.decoder_pos_embed.weight"], (cfg.chunk_size, d))
    layernorm(arena, sd, "model.decoder.norm")
    linear(arena, sd, "model.action_head", cfg.action_feature.shape[0], d)

    write_meta(out_dir, "impact", [
        f"dim {d}",
        f"heads {cfg.n_heads}",
        f"head_dim {d // cfg.n_heads}",
        f"ff {cfg.dim_feedforward}",
        f"n_enc {cfg.n_encoder_layers}",
        f"n_dec {cfg.n_decoder_layers}",
        f"chunk {cfg.chunk_size}",
        f"state_dim {state_dim}",
        f"action_dim {cfg.action_feature.shape[0]}",
        f"n_1d {n_1d}",
        f"n_text {cfg.tokenizer_max_length}",
        "ln_eps 1e-5",
    ])
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
            np.asarray(_to_numpy(v), np.float32).reshape(-1)[:n]

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
                 "tools/octo/convert_t5_tokenizer.py has the sentencepiece path")

    table = [None] * vocab_full
    for i, entry in enumerate(pieces):
        if i >= vocab_full:
            break
        table[i] = (entry[0], float(entry[1]))
    # The 100 extra_id sentinels live in added_tokens, past the unigram table.
    for i in range(len(pieces), vocab_full):
        table[i] = (fast.convert_ids_to_tokens(i), 0.0)

    with open(os.path.join(out_dir, "vocab.txt"), "w") as f:
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
    stats, want = {}, ("mean", "std")
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*normalizer*.safetensors"))):
        try:
            from safetensors import safe_open
        except ImportError:
            return {}
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                feat, _, field = k.rpartition(".")
                if field in want:
                    stats.setdefault(feat, {})[field] = _to_numpy(sf.get_tensor(k))
        if stats:
            break                      # preprocessor sorts first and holds them all
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
    ap.add_argument("--checkpoint", help="lerobot pretrained_model directory")
    ap.add_argument("--random", action="store_true", help="random weights at --seed")
    ap.add_argument("--out", default="build/impact")
    ap.add_argument("--img", default="480x640", help="HxW of the camera frames")
    ap.add_argument("--cams", default="observation.images.front,observation.images.wrist")
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--instruction", default="Put the tape into the box")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.checkpoint and not args.random:
        ap.error("pass --checkpoint DIR or --random")

    img_h, img_w = (int(x) for x in args.img.lower().split("x"))
    cam_keys = args.cams.split(",")
    os.makedirs(args.out, exist_ok=True)

    if args.checkpoint:
        from lerobot.policies.impact.modeling_impact import IMPACTPolicy
        log(f"loading {args.checkpoint}")
        policy = IMPACTPolicy.from_pretrained(args.checkpoint)
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
    log(f"converting -> {args.out}  ({img_h}x{img_w}, cams {cam_keys})")

    film_after = dump_vision(sd, policy, args.out)
    film_channels = policy.model.backbone.stage_channels if film_after else []
    vocab_full = dump_text(policy, args.out, film_channels)
    dump_transformer(sd, policy, args.out)

    state_dim = policy.config.robot_state_feature.shape[0]
    action_dim = policy.config.action_feature.shape[0]
    stats = load_dataset_stats(args.checkpoint)
    if stats:
        log(f"  stats           dataset statistics from the checkpoint's normalizer "
            f"({len(stats)} features)")
    else:
        log("  stats           IDENTITY - no normalizer found (random init?)")
    dump_stats(stats, cam_keys, args.out, state_dim, action_dim)
    tok = dump_tokenizer(args.out, vocab_full)
    dump_config(args.out, img_h, img_w, [k.split(".")[-1] for k in cam_keys],
                vocab_full, tok.unk_token_id if tok.unk_token_id is not None else 2,
                args.instruction)
    log(f"done -> {args.out}")


if __name__ == "__main__":
    main()
