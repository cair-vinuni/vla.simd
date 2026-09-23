#!/usr/bin/env python3
"""
convert_diffusion.py -- convert a lerobot Diffusion Policy checkpoint into the
flat .meta/.bin arenas the vla.simd engine loads.

    python3 tools/convert_diffusion.py \
        --ckpt ~/work/lerobot/outputs/train/dp_so101/checkpoints/last/pretrained_model \
        --out build/diffusion_so101

    # structural export with no trained weights, for bringing a port up:
    python3 tools/convert_diffusion.py --random-init --out build/diffusion_rand

Writes, into --out:

    diffusion.meta          shapes + sampler settings the C++ loader reads
    rgb_encoder{i}_backbone.{meta,bin}   ResNet-18, BatchNorm folded into the conv
    rgb_encoder{i}.bin      1x1 keypoint conv + the output projection
    unet.bin                the whole conditional UNet1d, in loader order
    stats.bin               state/action MIN-MAX and per-camera image MEAN/STD

**Random init is not validation of the model, only of the port.** It proves the
engine computes what the reference computes for these weights; it says nothing
about whether the checkpoint is any good, and -- the lesson IMPACT paid for --
it cannot catch a bug in the dataset statistics, because parity normalizes both
sides the same way. The engine refuses degenerate action stats at load for that
reason, and a random-init export writes deliberately non-degenerate ones so the
guard is exercised rather than bypassed.

The scheduler and step count are inference parameters, not weights: --scheduler
and --steps only change what goes in the meta, so one converted checkpoint
serves both the DDPM and DDIM rows.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# weight arena helpers (same cursor discipline as tools/act)
# ---------------------------------------------------------------------------
class Arena:
    """Append-only float32 blob, mirroring the `take()` cursor in the C++ loader."""

    def __init__(self):
        self.parts, self.n = [], 0

    def add(self, t):
        a = np.ascontiguousarray(_np(t), dtype=np.float32)
        self.parts.append(a.reshape(-1))
        self.n += a.size
        return self

    def write(self, path):
        blob = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
        blob.tofile(path)
        return blob.size


def _np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


def fold_bn(conv_w, sd, bn_prefix, eps=1e-5):
    """conv (no bias) + BatchNorm in eval -> conv weights and a bias."""
    gamma = sd[f"{bn_prefix}.weight"].to(torch.float32)
    beta = sd[f"{bn_prefix}.bias"].to(torch.float32)
    mean = sd[f"{bn_prefix}.running_mean"].to(torch.float32)
    var = sd[f"{bn_prefix}.running_var"].to(torch.float32)
    scale = gamma / torch.sqrt(var + eps)
    return conv_w.to(torch.float32) * scale.reshape(-1, 1, 1, 1), beta - mean * scale


def conv_nhwc(arena, w, b):
    """torch [Cout, Cin, k, k] -> engine [Cout, k, k, Cin]."""
    arena.add(w.permute(0, 2, 3, 1).contiguous()).add(b)


def conv1d_rows(arena, w, b):
    """torch Conv1d [Cout, Cin, k] -> engine [Cout, k*Cin], the im2col row order.

    The engine holds every 1-D conv as an nn::Linear over im2col'd taps, so the
    tap axis has to be the SLOW one within a row -- [Cout, k, Cin], not
    [Cout, Cin, k]. Getting this backwards produces a model that loads, runs, and
    is wrong, which is why the transpose lives here and is asserted on shape.
    """
    assert w.dim() == 3, f"expected Conv1d weight, got {tuple(w.shape)}"
    arena.add(w.permute(0, 2, 1).contiguous().reshape(w.shape[0], -1)).add(b)


def convT1d_rows(arena, w, b):
    """torch ConvTranspose1d [Cin, Cout, k] -> engine [Cin, k, Cout]."""
    assert w.dim() == 3, f"expected ConvTranspose1d weight, got {tuple(w.shape)}"
    arena.add(w.permute(0, 2, 1).contiguous()).add(b)


def linear(arena, sd, prefix):
    arena.add(sd[f"{prefix}.weight"]).add(sd[f"{prefix}.bias"])


# ---------------------------------------------------------------------------
# ResNet-18 backbone (nn.Sequential of torchvision children[:-2])
# ---------------------------------------------------------------------------
def dump_backbone(sd, prefix, out_dir, name):
    """The encoder's backbone is Sequential(conv1, bn1, relu, maxpool, layer1..4),
    so the stages sit at indices 4..7 rather than under .layerN names."""
    meta, arena = [], Arena()

    stem_w = sd[f"{prefix}.0.weight"]
    cout, cin, k, _ = stem_w.shape
    w, b = fold_bn(stem_w, sd, f"{prefix}.1")
    conv_nhwc(arena, w, b)
    meta += [f"in_ch {cin}", f"stem_out {cout}", f"stem_k {k}",
             "stem_stride 2", f"stem_pad {k // 2}",
             "pool_k 3", "pool_stride 2", "pool_pad 1"]

    for si, stage in enumerate(range(4, 8)):
        if f"{prefix}.{stage}.2.conv1.weight" in sd:
            sys.exit(f"{prefix}.{stage} has more than 2 blocks -- not a resnet18")
        for blk in range(2):
            p = f"{prefix}.{stage}.{blk}"
            if f"{p}.conv1.weight" not in sd:
                sys.exit(f"missing {p}.conv1.weight -- is this a resnet18 backbone?")
            c1 = sd[f"{p}.conv1.weight"]
            bcout, bcin, bk, _ = c1.shape
            assert bk == 3, f"{p}: basic blocks only, got k={bk}"
            stride = 2 if (si > 0 and blk == 0) else 1
            has_down = f"{p}.downsample.0.weight" in sd

            w, b = fold_bn(c1, sd, f"{p}.bn1")
            conv_nhwc(arena, w, b)
            w, b = fold_bn(sd[f"{p}.conv2.weight"], sd, f"{p}.bn2")
            conv_nhwc(arena, w, b)
            if has_down:
                w, b = fold_bn(sd[f"{p}.downsample.0.weight"], sd, f"{p}.downsample.1")
                conv_nhwc(arena, w, b)
            meta.append(f"block {bcin} {bcout} {stride} {int(has_down)}")

    with open(os.path.join(out_dir, f"{name}.meta"), "w") as f:
        f.write("\n".join(meta) + "\n")
    n = arena.write(os.path.join(out_dir, f"{name}.bin"))
    log(f"  {name}.bin {n * 4 / 1e6:.1f} MB")


def dump_rgb_encoder(sd, prefix, out_dir, name):
    dump_backbone(sd, f"{prefix}.backbone", out_dir, f"{name}_backbone")
    arena = Arena()
    # SpatialSoftmax's 1x1 conv, stored as a [K, C] linear.
    w = sd[f"{prefix}.pool.nets.weight"]
    assert w.shape[2] == 1 and w.shape[3] == 1, f"keypoint conv is not 1x1: {tuple(w.shape)}"
    arena.add(w.reshape(w.shape[0], w.shape[1])).add(sd[f"{prefix}.pool.nets.bias"])
    linear(arena, sd, f"{prefix}.out")
    n = arena.write(os.path.join(out_dir, f"{name}.bin"))
    log(f"  {name}.bin {n * 4 / 1e3:.1f} KB")


# ---------------------------------------------------------------------------
# UNet1d -- order must match DPUNet1d::load exactly
# ---------------------------------------------------------------------------
def dump_res_block(arena, sd, p, film_scale):
    conv1d_rows(arena, sd[f"{p}.conv1.block.0.weight"], sd[f"{p}.conv1.block.0.bias"])
    arena.add(sd[f"{p}.conv1.block.1.weight"]).add(sd[f"{p}.conv1.block.1.bias"])
    linear(arena, sd, f"{p}.cond_encoder.1")
    conv1d_rows(arena, sd[f"{p}.conv2.block.0.weight"], sd[f"{p}.conv2.block.0.bias"])
    arena.add(sd[f"{p}.conv2.block.1.weight"]).add(sd[f"{p}.conv2.block.1.bias"])
    if f"{p}.residual_conv.weight" in sd:
        w = sd[f"{p}.residual_conv.weight"]
        arena.add(w.reshape(w.shape[0], w.shape[1])).add(sd[f"{p}.residual_conv.bias"])


def dump_unet(sd, out_dir, cfg):
    arena = Arena()
    linear(arena, sd, "diffusion.unet.diffusion_step_encoder.1")
    linear(arena, sd, "diffusion.unet.diffusion_step_encoder.3")

    n_down = len(cfg.down_dims)
    for i in range(n_down):
        p = f"diffusion.unet.down_modules.{i}"
        dump_res_block(arena, sd, f"{p}.0", cfg.use_film_scale_modulation)
        dump_res_block(arena, sd, f"{p}.1", cfg.use_film_scale_modulation)
        if f"{p}.2.weight" in sd:                      # Conv1d, else Identity
            conv1d_rows(arena, sd[f"{p}.2.weight"], sd[f"{p}.2.bias"])

    dump_res_block(arena, sd, "diffusion.unet.mid_modules.0", cfg.use_film_scale_modulation)
    dump_res_block(arena, sd, "diffusion.unet.mid_modules.1", cfg.use_film_scale_modulation)

    for i in range(n_down - 1):
        p = f"diffusion.unet.up_modules.{i}"
        dump_res_block(arena, sd, f"{p}.0", cfg.use_film_scale_modulation)
        dump_res_block(arena, sd, f"{p}.1", cfg.use_film_scale_modulation)
        if f"{p}.2.weight" in sd:                      # ConvTranspose1d, else Identity
            convT1d_rows(arena, sd[f"{p}.2.weight"], sd[f"{p}.2.bias"])

    conv1d_rows(arena, sd["diffusion.unet.final_conv.0.block.0.weight"],
                sd["diffusion.unet.final_conv.0.block.0.bias"])
    arena.add(sd["diffusion.unet.final_conv.0.block.1.weight"]).add(
        sd["diffusion.unet.final_conv.0.block.1.bias"])
    w = sd["diffusion.unet.final_conv.1.weight"]
    arena.add(w.reshape(w.shape[0], w.shape[1])).add(sd["diffusion.unet.final_conv.1.bias"])

    n = arena.write(os.path.join(out_dir, "unet.bin"))
    log(f"  unet.bin {n * 4 / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# stats + meta
# ---------------------------------------------------------------------------
def dump_stats(stats, cam_keys, cfg, out_dir):
    """state/action MIN-MAX, then per-camera image MEAN/STD.

    DP normalizes state and action to [-1, 1] from the dataset min/max, which is
    what makes the sampler's clip_sample_range of 1.0 the right number. Writing
    MEAN/STD here instead would load, run, and command the wrong thing.
    """
    def get(key, field):
        s = stats.get(key)
        if s is None or field not in s:
            sys.exit(f"stats for {key!r} have no {field!r}")
        return np.asarray(_np(s[field]), dtype=np.float32).reshape(-1)

    parts = [get("observation.state", "min"), get("observation.state", "max"),
             get("action", "min"), get("action", "max")]
    for k in cam_keys:
        s = stats.get(k, {})
        mean = np.asarray(_np(s["mean"]), np.float32).reshape(-1) if "mean" in s \
            else np.zeros(3, np.float32)
        std = np.asarray(_np(s["std"]), np.float32).reshape(-1) if "std" in s \
            else np.ones(3, np.float32)
        parts += [mean[:3], std[:3]]

    span = parts[3] - parts[2]
    if not np.any(np.abs(span) > 1e-6):
        sys.exit("action min == max on every dimension: the checkpoint's normalizer "
                 "carries no dataset statistics. The engine rejects this at load "
                 "rather than command a robot from identity stats.")
    log(f"  stats action span {float(np.min(np.abs(span))):.4g} .. "
        f"{float(np.max(np.abs(span))):.4g}")
    np.concatenate(parts).astype(np.float32).tofile(os.path.join(out_dir, "stats.bin"))


def dump_meta(cfg, cam_names, img_hw, out_dir, scheduler, steps):
    h, w = img_hw
    crop = cfg.crop_shape or (0, 0)
    resize = cfg.resize_shape or (0, 0)
    lines = [
        f"n_obs_steps {cfg.n_obs_steps}",
        f"n_cams {len(cam_names)}",
        f"img_h {h}", f"img_w {w}",
        f"crop_h {crop[0]}", f"crop_w {crop[1]}",
        f"resize_h {resize[0]}", f"resize_w {resize[1]}",
        f"state_dim {cfg.robot_state_feature.shape[0]}",
        f"action_dim {cfg.action_feature.shape[0]}",
        f"horizon {cfg.horizon}",
        f"n_action_steps {cfg.n_action_steps}",
        f"num_keypoints {cfg.spatial_softmax_num_keypoints}",
        f"kernel_size {cfg.kernel_size}",
        f"n_groups {cfg.n_groups}",
        f"step_embed_dim {cfg.diffusion_step_embed_dim}",
        f"film_scale {int(cfg.use_film_scale_modulation)}",
        f"separate_encoders {int(cfg.use_separate_rgb_encoder_per_camera)}",
        "down_dims " + " ".join(str(d) for d in cfg.down_dims),
        f"num_train_timesteps {cfg.num_train_timesteps}",
        f"num_inference_steps {steps}",
        f"beta_start {cfg.beta_start}",
        f"beta_end {cfg.beta_end}",
        f"beta_schedule {cfg.beta_schedule}",
        f"prediction_type {cfg.prediction_type}",
        f"clip_sample {int(cfg.clip_sample)}",
        f"clip_sample_range {cfg.clip_sample_range}",
        f"scheduler {scheduler}",
        "gn_eps 1e-5",
    ] + [f"cam {n}" for n in cam_names]
    with open(os.path.join(out_dir, "diffusion.meta"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        f.write("".join(f"cam{i} {n.rsplit('.', 1)[-1]}\n" for i, n in enumerate(cam_names)))


# ---------------------------------------------------------------------------
def build_random(args):
    """A structurally real DP with untrained weights, for bringing the port up.

    Shapes default to the SO-101 observation setup the other policies in this
    campaign use (2 cameras, 480x640, 6-DoF state and action), so the port is
    exercised at the size it will actually run at.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    h, w = args.img_h, args.img_w
    cams = [f"observation.images.cam{i}" for i in range(args.n_cams)]
    down_dims = tuple(int(x) for x in args.down_dims.split(",")) if args.down_dims \
        else (512, 1024, 2048)
    cfg = DiffusionConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        down_dims=down_dims,
        crop_shape=(args.crop_h, args.crop_w) if args.crop_h else None,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
            **{c: PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)) for c in cams},
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,)),
        },
    )
    policy = DiffusionPolicy(cfg)
    policy.eval()

    # Non-degenerate statistics on purpose: identity stats are the failure the
    # engine refuses at load, and an export that wrote them would make that
    # guard untestable.
    rng = np.random.default_rng(0)
    stats = {
        "observation.state": {
            "min": np.full(args.state_dim, -100.0, np.float32),
            "max": np.full(args.state_dim, 100.0, np.float32),
        },
        "action": {
            "min": np.full(args.action_dim, -100.0, np.float32),
            "max": np.full(args.action_dim, 100.0, np.float32),
        },
    }
    for c in cams:
        stats[c] = {"mean": rng.uniform(0.4, 0.6, 3).astype(np.float32),
                    "std": rng.uniform(0.2, 0.3, 3).astype(np.float32)}
    return policy, cfg, cams, (h, w), stats


def load_checkpoint(path):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(path)
    policy.eval()
    policy.to("cpu")
    cfg = policy.config
    cams = list(cfg.image_features.keys())
    shape = next(iter(cfg.image_features.values())).shape
    stats = _checkpoint_stats(path, cfg)
    return policy, cfg, cams, (shape[1], shape[2]), stats


def _checkpoint_stats(path, cfg):
    """A trained checkpoint carries dataset statistics in its preprocessor.

    lerobot writes them as a safetensors blob of flat `{feature}.{stat}` entries
    beside the processor's JSON, not inside it -- so the JSON paths below are the
    fallback for older checkpoints, and the safetensors is the one that fires.
    """
    import glob

    for f in sorted(glob.glob(os.path.join(path, "*normalizer_processor*.safetensors"))):
        from safetensors.numpy import load_file
        blob = load_file(f)
        out = {}
        for k, v in blob.items():
            feat, stat = k.rsplit(".", 1)
            out.setdefault(feat, {})[stat] = np.asarray(v, np.float32)
        if "action" in out:
            return out

    for name in ("preprocessor_config.json", "config.json"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            with open(p) as f:
                blob = json.load(f)
            st = blob.get("dataset_stats") or blob.get("stats")
            if st:
                return {k: {kk: np.asarray(vv, np.float32) for kk, vv in v.items()}
                        for k, v in st.items()}
    for p in glob.glob(os.path.join(path, "**", "*stats*.json"), recursive=True):
        with open(p) as f:
            blob = json.load(f)
        if isinstance(blob, dict) and "action" in blob:
            return {k: {kk: np.asarray(vv, np.float32) for kk, vv in v.items()}
                    for k, v in blob.items()}
    sys.exit(f"no dataset statistics found under {path}. Exporting without them would "
             "produce a model that normalizes with identity stats -- refusing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheduler", default=None, choices=["DDPM", "DDIM"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # random-init shape knobs; ignored with --ckpt
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--n-cams", type=int, default=2)
    ap.add_argument("--img-h", type=int, default=480)
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--crop-h", type=int, default=0)
    ap.add_argument("--crop-w", type=int, default=0)
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--n-action-steps", type=int, default=32)
    # The stock UNet is ~256M parameters and a 1 GB arena regardless of image
    # size. Narrowing it keeps every code path (down/mid/up, FiLM, the skip
    # concat) and shrinks the blob.
    ap.add_argument("--down-dims", default=None,
                    help="comma-separated UNet channel widths, e.g. 64,128,256")
    args = ap.parse_args()

    if bool(args.ckpt) == bool(args.random_init):
        sys.exit("give exactly one of --ckpt or --random-init")

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.random_init:
        log("building a random-init DiffusionPolicy (port validation only)")
        policy, cfg, cams, img_hw, stats = build_random(args)
    else:
        log(f"loading {args.ckpt}")
        policy, cfg, cams, img_hw, stats = load_checkpoint(args.ckpt)

    if cfg.resize_shape and tuple(cfg.resize_shape) != tuple(img_hw):
        sys.exit(f"resize_shape {tuple(cfg.resize_shape)} != image size {tuple(img_hw)}: "
                 "the engine does not resize")
    if cfg.use_group_norm:
        sys.exit("use_group_norm=True: the engine folds BatchNorm running stats into the "
                 "convs, and a GroupNorm backbone has none")
    if cfg.horizon < cfg.n_obs_steps - 1 + cfg.n_action_steps:
        sys.exit(f"horizon {cfg.horizon} < n_obs_steps - 1 + n_action_steps "
                 f"({cfg.n_obs_steps} - 1 + {cfg.n_action_steps})")
    want = {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    got = {k: cfg.normalization_mapping.get(k) for k in want}
    if got != want:
        sys.exit(f"normalization_mapping {got} != {want}, which the engine hard-codes")

    scheduler = args.scheduler or ("DDIM" if cfg.noise_scheduler_type == "DDIM" else "DDPM")
    steps = args.steps or cfg.num_inference_steps or cfg.num_train_timesteps
    if steps > cfg.num_train_timesteps:
        sys.exit(f"--steps {steps} exceeds the trained {cfg.num_train_timesteps}")

    sd = policy.state_dict()
    log(f"config: {cfg.n_obs_steps} obs x {len(cams)} cams at {img_hw[0]}x{img_hw[1]}, "
        f"horizon {cfg.horizon}, execute {cfg.n_action_steps}, {scheduler} x{steps}")

    n_enc = len(cams) if cfg.use_separate_rgb_encoder_per_camera else 1
    for i in range(n_enc):
        prefix = f"diffusion.rgb_encoder.{i}" if cfg.use_separate_rgb_encoder_per_camera \
            else "diffusion.rgb_encoder"
        dump_rgb_encoder(sd, prefix, args.out, f"rgb_encoder{i}")

    dump_unet(sd, args.out, cfg)
    dump_stats(stats, cams, cfg, args.out)
    dump_meta(cfg, cams, img_hw, args.out, scheduler, steps)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
