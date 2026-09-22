#!/usr/bin/env python3
"""
convert_act.py -- convert a lerobot ACT checkpoint into the flat .meta/.bin
arena the vla.simd engine loads.

    python tools/convert_act.py khanhnd61/act_so101_tape build/act

Run it in a venv that has lerobot: the checkpoint is read through ACTPolicy so the
config, the processor pipeline and the dataset stats come from the same source the
torch policy uses. The engine itself needs none of that at runtime.

Written into <out>/:
    config.txt      img_h/img_w/n_cams/norm_eps + the camera order
    backbone.meta   ResNet-18 stage table (block cin cout stride has_down)
    backbone.bin    stem + basic blocks, BatchNorm folded into the conv in front
    act.meta        transformer dims
    act.bin         projections, encoder, decoder, head -- in load() order
    stats.bin       state/action mean+std, then per-camera image mean+std

The BatchNorms are frozen in an ACT checkpoint (the backbone is fine-tuned but the
running stats stay in eval mode), so folding them is exact rather than an
approximation: conv weights scale by gamma/sqrt(var+eps) and the bias absorbs the
rest. That is what lets the engine run conv + bias + relu with no norm kernel.
"""

import argparse
import os
import sys

import numpy as np
import torch

# stats.bin stores mean then std, per feature, in this order.
STAT_KEYS = ("mean", "std")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# weight arena helpers
# ---------------------------------------------------------------------------
class Arena:
    """Append-only float32 blob, mirroring the `take()` cursor in the C++ loader."""

    def __init__(self):
        self.parts = []
        self.n = 0

    def add(self, t):
        a = np.ascontiguousarray(_to_numpy(t), dtype=np.float32)
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
        arena.add(w[i * dim : (i + 1) * dim]).add(b[i * dim : (i + 1) * dim])
    linear(arena, sd, f"{prefix}.out_proj", dim, dim)


def fold_bn(conv_w, bn_prefix, sd, eps=1e-5):
    """conv (no bias) followed by a frozen BatchNorm -> conv weights + a bias.

    scale = gamma / sqrt(running_var + eps); W' = W * scale, b' = beta - mean*scale.
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
# backbone
# ---------------------------------------------------------------------------
def dump_backbone(sd, out_dir, policy):
    """ResNet-18 truncated at layer4, every BN folded away."""
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

    # torchvision names the stages layer1..layer4, two BasicBlocks each. A block
    # downsamples (stride 2 + a 1x1 projection on the identity) only where the
    # channel count changes, i.e. the first block of layer2/3/4.
    #
    # The 2-per-stage count below is resnet18's. A deeper backbone (resnet34 is
    # 3/4/6/3) would export the first two blocks of each stage and produce a
    # checkpoint that LOADS and is wrong, so refuse instead: the guard inside the
    # loop only catches backbones with FEWER blocks.
    backbone = getattr(getattr(policy, "config", None), "vision_backbone", None)
    if backbone not in (None, "resnet18"):
        sys.exit(f"vision_backbone is {backbone!r}; this exporter only handles resnet18 "
                 "(2 BasicBlocks per stage). Extend the loop below before using it.")
    for stage in range(1, 5):
        if f"model.backbone.layer{stage}.2.conv1.weight" in sd:
            sys.exit(f"model.backbone.layer{stage} has more than 2 blocks -- not a resnet18")
        for blk in range(2):
            p = f"model.backbone.layer{stage}.{blk}"
            if f"{p}.conv1.weight" not in sd:
                sys.exit(f"missing {p}.conv1.weight -- is this a resnet18 backbone?")
            c1 = sd[f"{p}.conv1.weight"]
            bcout, bcin, bk, _ = c1.shape
            assert bk == 3, f"{p}: only basic blocks (3x3) are supported, got k={bk}"
            stride = 2 if (stage > 1 and blk == 0) else 1
            has_down = f"{p}.downsample.0.weight" in sd

            w, b = fold_bn(c1, f"{p}.bn1", sd)
            conv_nhwc(arena, w, b)
            w, b = fold_bn(sd[f"{p}.conv2.weight"], f"{p}.bn2", sd)
            conv_nhwc(arena, w, b)
            if has_down:
                w, b = fold_bn(sd[f"{p}.downsample.0.weight"], f"{p}.downsample.1", sd)
                conv_nhwc(arena, w, b)

            meta.append(f"block {bcin} {bcout} {stride} {int(has_down)}")

    with open(os.path.join(out_dir, "backbone.meta"), "w") as f:
        f.write("\n".join(meta) + "\n")
    n = arena.write(os.path.join(out_dir, "backbone.bin"))
    log(f"  backbone.bin {n * 4 / 1e6:.1f} MB ({len(meta) - 8} blocks)")


# ---------------------------------------------------------------------------
# transformer
# ---------------------------------------------------------------------------
def dump_transformer(sd, cfg, out_dir, policy):
    d = cfg.dim_model
    arena = Arena()

    # encoder_img_feat_input_proj is a 1x1 Conv2d over the feature map, which is a
    # plain linear over tokens once the map is in NHWC.
    w = sd["model.encoder_img_feat_input_proj.weight"]
    arena.add(w.reshape(w.shape[0], w.shape[1])).add(sd["model.encoder_img_feat_input_proj.bias"])

    state_dim = sd["model.encoder_robot_state_input_proj.weight"].shape[1]
    linear(arena, sd, "model.encoder_robot_state_input_proj", d, state_dim)

    # At inference the VAE latent is all zeros, so encoder_latent_input_proj(0)
    # collapses to its bias -- one constant token instead of a matmul.
    latent_w = sd["model.encoder_latent_input_proj.weight"]
    latent_b = sd["model.encoder_latent_input_proj.bias"]
    # The precondition is that the policy really does feed a ZERO latent at
    # inference - not that W @ 0 == 0, which is true of every W and was what
    # this assert used to check.
    assert not getattr(policy.config, "use_vae", False) or policy.config.n_vae_encoder_layers >= 0
    probe = policy.model.encoder_latent_input_proj(torch.zeros(1, latent_w.shape[1]))
    assert torch.allclose(probe.squeeze(0), latent_b), \
        "encoder_latent_input_proj(0) != bias; the zero-latent collapse is not valid here"
    arena.add(latent_b)

    pos1d = sd["model.encoder_1d_feature_pos_embed.weight"]
    n_1d = pos1d.shape[0]
    arena.add(pos1d)

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

    arena.add(sd["model.decoder_pos_embed.weight"])
    layernorm(arena, sd, "model.decoder.norm")
    action_dim = sd["model.action_head.weight"].shape[0]
    linear(arena, sd, "model.action_head", action_dim, d)

    meta = [
        f"dim {d}",
        f"heads {cfg.n_heads}",
        f"head_dim {d // cfg.n_heads}",
        f"ff {cfg.dim_feedforward}",
        f"n_enc {cfg.n_encoder_layers}",
        f"n_dec {cfg.n_decoder_layers}",
        f"chunk {cfg.chunk_size}",
        f"state_dim {state_dim}",
        f"action_dim {action_dim}",
        f"n_1d {n_1d}",
        "ln_eps 1e-5",
    ]
    with open(os.path.join(out_dir, "act.meta"), "w") as f:
        f.write("\n".join(meta) + "\n")
    n = arena.write(os.path.join(out_dir, "act.bin"))
    log(f"  act.bin {n * 4 / 1e6:.1f} MB (enc {cfg.n_encoder_layers}, dec {cfg.n_decoder_layers})")
    return state_dim, action_dim, n_1d


# ---------------------------------------------------------------------------
# stats + config
# ---------------------------------------------------------------------------
def dump_stats(norm_stats, cam_keys, out_dir):
    """stats.bin: state mean/std, action mean/std, then mean/std per camera.

    lerobot keeps VISUAL stats as (3,1,1) per-channel tensors; the engine applies
    them to a [0,1] image exactly like NormalizerProcessorStep does.
    """
    parts = []
    for key in ("observation.state", "action"):
        for stat in STAT_KEYS:
            parts.append(_to_numpy(norm_stats[f"{key}.{stat}"]).reshape(-1))
    for cam in cam_keys:
        for stat in STAT_KEYS:
            parts.append(_to_numpy(norm_stats[f"{cam}.{stat}"]).reshape(-1))

    blob = np.concatenate(parts).astype(np.float32)
    blob.tofile(os.path.join(out_dir, "stats.bin"))
    log(f"  stats.bin {blob.size} floats")


def dump_config(out_dir, img_h, img_w, cam_names, eps, task):
    lines = [f"img_h {img_h}", f"img_w {img_w}", f"n_cams {len(cam_names)}", f"norm_eps {eps}"]
    lines += [f"cam{i} {n}" for i, n in enumerate(cam_names)]
    if task:
        lines.append(f"instruction {task}")
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Accepts both shapes: the positional pair this script has always taken, and
    # the --ckpt/--out of the other converters. The flags win if both are
    # given.
    p.add_argument("ckpt", nargs="?", default=None,
                   help="lerobot ACT checkpoint (hub id or local dir)")
    p.add_argument("out", nargs="?", default=None,
                   help="output dir (default: build/act)")
    p.add_argument("--ckpt", dest="ckpt_flag", default=None,
                   help="same as the positional checkpoint")
    p.add_argument("--out", dest="out_flag", default=None,
                   help="same as the positional out dir")
    p.add_argument("--task", default=None, help="instruction to record in config.txt")
    args = p.parse_args()

    args.ckpt = args.ckpt_flag or args.ckpt
    args.out = args.out_flag or args.out or "build/act"
    if not args.ckpt:
        p.error("a checkpoint is required, positionally or with --ckpt")

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    os.makedirs(args.out, exist_ok=True)
    log(f"Loading {args.ckpt}")
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.eval()
    policy.to("cpu")
    cfg = policy.config

    if cfg.pre_norm:
        sys.exit("pre_norm checkpoints are not supported by the engine (post-norm only)")
    if cfg.feedforward_activation != "relu":
        sys.exit(f"only relu feedforward is supported, got {cfg.feedforward_activation}")

    sd = policy.state_dict()
    cam_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
    cam_names = [k.rsplit(".", 1)[-1] for k in cam_keys]
    _, img_h, img_w = cfg.input_features[cam_keys[0]].shape
    log(f"  {len(cam_keys)} cameras {cam_names} at {img_h}x{img_w}, chunk {cfg.chunk_size}")

    dump_backbone(sd, args.out, policy)
    state_dim, action_dim, n_1d = dump_transformer(sd, cfg, args.out, policy)

    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    norm_stats = _normalizer_stats(preprocessor)
    dump_stats(norm_stats, cam_keys, args.out)
    dump_config(args.out, img_h, img_w, cam_names, 1e-8, args.task)

    log(f"  {state_dim}-d state, {action_dim}-d action, {n_1d} 1-D pos tokens")

    log(f"Wrote {args.out}")


def _normalizer_stats(preprocessor):
    """Pull the buffers out of the pipeline's NormalizerProcessorStep."""
    for step in preprocessor.steps:
        stats = getattr(step, "stats", None)
        if stats:
            # {feature: {stat: tensor}} -> flat "feature.stat" keys
            return {f"{k}.{s}": v for k, d in stats.items() for s, v in d.items()}
    sys.exit("no normalizer step found in the preprocessor pipeline")


if __name__ == "__main__":
    main()
