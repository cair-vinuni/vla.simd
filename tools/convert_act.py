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

from _common import Arena, dump_detr, dump_resnet18, log, to_numpy, write_meta

# stats.bin stores mean then std, per feature, in this order.
STAT_KEYS = ("mean", "std")


# ---------------------------------------------------------------------------
# backbone
# ---------------------------------------------------------------------------
def dump_backbone(sd, out_dir, policy):
    """ResNet-18 truncated at layer4, every BN folded away."""
    backbone = getattr(getattr(policy, "config", None), "vision_backbone", None)
    if backbone not in (None, "resnet18"):
        sys.exit(f"vision_backbone is {backbone!r}; this exporter only handles resnet18 "
                 "(2 BasicBlocks per stage). Extend the loop below before using it.")
    arena = Arena()
    meta = dump_resnet18(arena, sd, "model.backbone.conv1", "model.backbone.bn1",
                         [f"model.backbone.layer{i}" for i in range(1, 5)])
    write_meta(out_dir, "backbone", meta)
    n = arena.write(os.path.join(out_dir, "backbone.bin"))
    log(f"  backbone.bin {n * 4 / 1e6:.1f} MB ({len(meta) - 8} blocks)")


# ---------------------------------------------------------------------------
# transformer
# ---------------------------------------------------------------------------
def dump_transformer(sd, cfg, out_dir):
    state_dim = sd["model.encoder_robot_state_input_proj.weight"].shape[1]
    action_dim = sd["model.action_head.weight"].shape[0]
    arena = Arena()
    meta = dump_detr(arena, sd, cfg, state_dim, action_dim)
    write_meta(out_dir, "act", meta + ["ln_eps 1e-5"])
    n = arena.write(os.path.join(out_dir, "act.bin"))
    log(f"  act.bin {n * 4 / 1e6:.1f} MB (enc {cfg.n_encoder_layers}, dec {cfg.n_decoder_layers})")
    return state_dim, action_dim, sd["model.encoder_1d_feature_pos_embed.weight"].shape[0]


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
            parts.append(to_numpy(norm_stats[f"{key}.{stat}"]).reshape(-1))
    for cam in cam_keys:
        for stat in STAT_KEYS:
            parts.append(to_numpy(norm_stats[f"{cam}.{stat}"]).reshape(-1))

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
    if cfg.env_state_feature or not cfg.robot_state_feature:
        sys.exit("ACT export needs a robot state and no environment state")
    want = dict.fromkeys(("VISUAL", "STATE", "ACTION"), "MEAN_STD")
    got = {k: cfg.normalization_mapping.get(k) for k in want}
    if got != want:
        sys.exit(f"normalization_mapping {got} != {want}, which the engine hard-codes")

    sd = policy.state_dict()
    cam_keys = list(cfg.image_features)
    cam_names = [k.rsplit(".", 1)[-1] for k in cam_keys]
    _, img_h, img_w = cfg.image_features[cam_keys[0]].shape
    log(f"  {len(cam_keys)} cameras {cam_names} at {img_h}x{img_w}, chunk {cfg.chunk_size}")

    dump_backbone(sd, args.out, policy)
    state_dim, action_dim, n_1d = dump_transformer(sd, cfg, args.out)

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
