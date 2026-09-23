import os
import sys

import numpy as np
import torch


def log(msg):
    print(msg, flush=True)


class Arena:
    """Append-only float32 blob, mirroring the `take()` cursor in the C++ loader."""

    def __init__(self):
        self.parts = []
        self.n = 0

    def add(self, t, shape=None):
        a = np.ascontiguousarray(to_numpy(t), dtype=np.float32)
        if shape is not None:
            assert a.shape == tuple(shape), f"expected {tuple(shape)}, got {a.shape}"
        self.parts.append(a.reshape(-1))
        self.n += a.size
        return self

    def write(self, path):
        blob = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
        blob.tofile(path)
        return blob.size


def to_numpy(t):
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


def dump_resnet18(arena, sd, stem_conv, stem_bn, stages, gn_group_size=0):
    def conv_norm(w, norm):
        if gn_group_size:
            arena.add(w.permute(0, 2, 3, 1).contiguous())
            arena.add(sd[f"{norm}.weight"]).add(sd[f"{norm}.bias"])
        else:
            conv_nhwc(arena, *fold_bn(w, norm, sd))

    stem_w = sd[f"{stem_conv}.weight"]
    cout, cin, k, _ = stem_w.shape
    conv_norm(stem_w, stem_bn)
    meta = [
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
    for si, stage in enumerate(stages):
        if f"{stage}.2.conv1.weight" in sd:
            sys.exit(f"{stage} has more than 2 blocks -- not a resnet18")
        for blk in range(2):
            p = f"{stage}.{blk}"
            if f"{p}.conv1.weight" not in sd:
                sys.exit(f"missing {p}.conv1.weight -- is this a resnet18 backbone?")
            c1 = sd[f"{p}.conv1.weight"]
            bcout, bcin, bk, _ = c1.shape
            assert bk == 3, f"{p}: only basic blocks (3x3) are supported, got k={bk}"
            stride = 2 if (si > 0 and blk == 0) else 1
            has_down = f"{p}.downsample.0.weight" in sd

            conv_norm(c1, f"{p}.bn1")
            conv_norm(sd[f"{p}.conv2.weight"], f"{p}.bn2")
            if has_down:
                conv_norm(sd[f"{p}.downsample.0.weight"], f"{p}.downsample.1")

            meta.append(f"block {bcin} {bcout} {stride} {int(has_down)}")
    if gn_group_size:
        meta.append(f"gn_group_size {gn_group_size}")
    return meta


def dump_detr(arena, sd, cfg, state_dim, action_dim):
    d = cfg.dim_model

    # encoder_img_feat_input_proj is a 1x1 Conv2d over the feature map, which is a
    # plain linear over tokens once the map is in NHWC.
    w = sd["model.encoder_img_feat_input_proj.weight"]
    assert w.shape[2:] == (1, 1), f"img proj is {tuple(w.shape)}, expected a 1x1 conv"
    arena.add(w.reshape(w.shape[0], w.shape[1])).add(sd["model.encoder_img_feat_input_proj.bias"])

    linear(arena, sd, "model.encoder_robot_state_input_proj", d, state_dim)

    # At inference the VAE latent is all zeros, so encoder_latent_input_proj(0)
    # collapses to its bias -- one constant token instead of a matmul.
    assert tuple(sd["model.encoder_latent_input_proj.weight"].shape) == (d, cfg.latent_dim)
    arena.add(sd["model.encoder_latent_input_proj.bias"], (d,))

    pos1d = sd["model.encoder_1d_feature_pos_embed.weight"]
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

    arena.add(sd["model.decoder_pos_embed.weight"], (cfg.chunk_size, d))
    layernorm(arena, sd, "model.decoder.norm")
    linear(arena, sd, "model.action_head", action_dim, d)

    return [
        f"dim {d}",
        f"heads {cfg.n_heads}",
        f"head_dim {d // cfg.n_heads}",
        f"ff {cfg.dim_feedforward}",
        f"n_enc {cfg.n_encoder_layers}",
        f"n_dec {cfg.n_decoder_layers}",
        f"chunk {cfg.chunk_size}",
        f"state_dim {state_dim}",
        f"action_dim {action_dim}",
        f"n_1d {pos1d.shape[0]}",
    ]
