#!/usr/bin/env python3
"""
convert_lerobot_octo.py

Convert a finetuned lerobot Octo checkpoint into the flat .meta/.bin weight format
of the vla.simd engine.

    python3 tools/octo/convert_lerobot_octo.py \
        --checkpoint khanhnd61/octo_so101_tape \
        --t5-from build/octo --out build/octo_so101

Why this exists alongside convert_octo.py: that script converts the authoritative
JAX/flax Octo-Small-1.5, the *base* model. An
SO-101 finetune produced through lerobot is a torch checkpoint with a different
parameter tree and, more importantly, three implementation differences that are
invisible unless reproduced deliberately:

  * **Weight standardization.** JAX Octo divides by `std + 1e-5`; octo-pytorch (and
    therefore lerobot) divides by `sqrt(var + 1e-10)`. The finetune was trained
    through the latter, so the stems are folded that way here. Using the JAX form
    injects a systematic per-channel error into every stem conv with no local
    symptom.
  * **GroupNorm epsilon.** flax defaults to 1e-6, `nn.GroupNorm` to 1e-5. The stem
    meta therefore records 1e-5 for these checkpoints, not the 1e-6 the JAX dump
    writes. The engine reads gn_eps from the meta, so this is a data change only.
  * **LayerNorm epsilon** is explicitly 1e-6 in lerobot's transformer, matching
    flax, so `ln_eps` is unchanged.

The language tower is frozen (`freeze_language_encoder=True`) and absent from the
checkpoint, so t5.meta/t5.bin and the tokenizer are copied from an existing base
dump rather than reconverted -- pass it with --t5-from.

Run in a venv with lerobot + torch. Nothing here touches a GPU.
"""

import argparse
import os
import shutil
import sys

import numpy as np

try:
    import torch
    import importlib
    importlib.import_module("lerobot.policies.octo.configuration_octo")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
except ImportError as e:  # pragma: no cover
    sys.exit(f"needs lerobot + torch in this interpreter: {e}")


def f32(x):
    """tensor/array -> contiguous fp32 numpy."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def std_conv(w):
    """Fold octo-pytorch weight standardization into a conv kernel.

    torch layout [Cout, Cin, kh, kw], normalized per output channel over the other
    three axes. `sqrt(var + 1e-10)` is octo-pytorch's form and differs from JAX's
    `std + 1e-5`; see the module docstring.
    """
    w = w.detach().cpu().to(torch.float64)
    var, mean = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    return ((w - mean) / torch.sqrt(var + 1e-10)).to(torch.float32)


def dump_stem(sd, prefix, out, name):
    """SmallStem16: 4 x (StdConv + GroupNorm) then a 1x1 patch embedding."""
    feats = []
    with open(f"{out}/{name}.bin", "wb") as f:
        for i in range(4):
            cw = std_conv(sd[f"{prefix}.layers.{i}.0.weight"])       # [Cout,Cin,3,3]
            feats.append(int(cw.shape[0]))
            # engine wants [Cout, kh, kw, Cin]
            f32(cw.permute(0, 2, 3, 1)).tofile(f)
            f32(sd[f"{prefix}.layers.{i}.0.bias"]).tofile(f)
            f32(sd[f"{prefix}.layers.{i}.1.weight"]).tofile(f)       # GroupNorm scale
            f32(sd[f"{prefix}.layers.{i}.1.bias"]).tofile(f)
        ew = sd[f"{prefix}.embedding.weight"]                        # [512,384,1,1]
        embed_dim = int(ew.shape[0])
        f32(ew.reshape(ew.shape[0], ew.shape[1])).tofile(f)          # [512,384] out,in
        f32(sd[f"{prefix}.embedding.bias"]).tofile(f)
    in_ch = int(sd[f"{prefix}.layers.0.0.weight"].shape[1])
    with open(f"{out}/{name}.meta", "w") as f:
        f.write(f"in_ch {in_ch}\nn_layers 4\nk 3\nstride 2\npad 1\n")
        f.write("features " + " ".join(map(str, feats)) + "\n")
        # nn.GroupNorm's default eps, not flax's 1e-6 -- see the module docstring.
        f.write(f"embed_dim {embed_dim}\ngn_groups 32\ngn_eps 1e-5\n")
    print(f"  {name}: in_ch={in_ch} features={feats} embed={embed_dim}")
    return in_ch


def dump_transformer(sd, cfg, out, n_layers, d, n_heads, mlp, window):
    ot = "model.octo_transformer"
    bt = f"{ot}.block_transformer.transformer"
    head_dim = d // n_heads
    with open(f"{out}/octo.meta", "w") as f:
        f.write(f"d {d}\nn_layers {n_layers}\nheads {n_heads}\nhead_dim {head_dim}\nmlp {mlp}\n")
        f.write(f"max_horizon {cfg.max_horizon}\nn_task {cfg.tokenizer_max_length}\n")
        f.write("tok_primary 256\ntok_wrist 64\nn_readout 1\n")
        f.write(f"t5_dim 768\nstem_dim 512\nln_eps 1e-6\nwindow {window}\n")
        # lerobot's transformer uses F.gelu (exact erf); flax's nn.gelu, which the
        # JAX dump matches, is the tanh approximation. Wrong form here is a small
        # systematic offset in every MLP, not an obvious failure.
        f.write("gelu_erf 1\n")
    with open(f"{out}/octo.bin", "wb") as f:
        tp = f"{ot}.task_projections.task_language_projection"
        op = f"{ot}.obs_projections"
        f32(sd[f"{tp}.weight"]).tofile(f)                            # [384,768] out,in
        f32(sd[f"{tp}.bias"]).tofile(f)
        f32(sd[f"{op}.obs_primary_projection.weight"]).tofile(f)     # [384,512]
        f32(sd[f"{op}.obs_primary_projection.bias"]).tofile(f)
        f32(sd[f"{op}.obs_wrist_projection.weight"]).tofile(f)
        f32(sd[f"{op}.obs_wrist_projection.bias"]).tofile(f)
        f32(sd[f"{ot}.task_language_pos_embedding"]).tofile(f)       # [16,384]
        f32(sd[f"{ot}.obs_primary_pos_embedding"]).tofile(f)         # [10,256,384]
        f32(sd[f"{ot}.obs_wrist_pos_embedding"]).tofile(f)           # [10,64,384]
        f32(sd[f"{ot}.readout_action_pos_embedding"]).tofile(f)      # [10,1,384]
        for L in range(n_layers):
            b = f"{bt}.encoder_blocks.{L}"
            f32(sd[f"{b}.layer_norm1.weight"]).tofile(f)
            f32(sd[f"{b}.layer_norm1.bias"]).tofile(f)
            # nn.MultiheadAttention fuses q,k,v into one [3d, d] projection; the
            # engine wants them separately, in q,k,v order.
            inw = sd[f"{b}.self_attention.in_proj_weight"]           # [1152,384]
            inb = sd[f"{b}.self_attention.in_proj_bias"]             # [1152]
            for i in range(3):
                f32(inw[i * d:(i + 1) * d]).tofile(f)                # [384,384] out,in
                f32(inb[i * d:(i + 1) * d]).tofile(f)
            f32(sd[f"{b}.self_attention.out_proj.weight"]).tofile(f)
            f32(sd[f"{b}.self_attention.out_proj.bias"]).tofile(f)
            f32(sd[f"{b}.layer_norm2.weight"]).tofile(f)
            f32(sd[f"{b}.layer_norm2.bias"]).tofile(f)
            f32(sd[f"{b}.mlp_block.dense1.weight"]).tofile(f)        # [1536,384]
            f32(sd[f"{b}.mlp_block.dense1.bias"]).tofile(f)
            f32(sd[f"{b}.mlp_block.dense2.weight"]).tofile(f)        # [384,1536]
            f32(sd[f"{b}.mlp_block.dense2.bias"]).tofile(f)
        f32(sd[f"{bt}.layer_norm.weight"]).tofile(f)
        f32(sd[f"{bt}.layer_norm.bias"]).tofile(f)
    print(f"  octo: d={d} layers={n_layers} heads={n_heads} mlp={mlp} window={window}")


def dump_head(sd, head, out, cfg):
    dm = "model.heads.action.diffusion_model"
    rn = f"{dm}.reverse_network"
    action_dim = int(sd[f"{rn}.linear2.weight"].shape[0]) // cfg.chunk_size
    with open(f"{out}/head.meta", "w") as f:
        f.write(f"emb 384\naction_dim {action_dim}\nhorizon {cfg.chunk_size}\ntime_dim 32\n")
        f.write(f"num_blocks 3\nhidden 256\nsteps {cfg.num_diffusion_steps}\n")
        f.write(f"max_action {float(cfg.max_action)}\n")
    with open(f"{out}/head.bin", "wb") as f:
        f32(sd[f"{dm}.time_preprocess.w"]).tofile(f)                 # [16,1]
        f32(sd[f"{dm}.cond_encoder.layers.0.weight"]).tofile(f)      # [64,32]
        f32(sd[f"{dm}.cond_encoder.layers.0.bias"]).tofile(f)
        f32(sd[f"{dm}.cond_encoder.layers.2.weight"]).tofile(f)      # [32,64]
        f32(sd[f"{dm}.cond_encoder.layers.2.bias"]).tofile(f)
        f32(sd[f"{rn}.linear1.weight"]).tofile(f)                    # [256, 384+flat+32]
        f32(sd[f"{rn}.linear1.bias"]).tofile(f)
        for i in range(3):
            b = f"{rn}.blocks.{i}"
            f32(sd[f"{b}.layer_norm.weight"]).tofile(f)
            f32(sd[f"{b}.layer_norm.bias"]).tofile(f)
            f32(sd[f"{b}.linear1.weight"]).tofile(f)                 # [1024,256]
            f32(sd[f"{b}.linear1.bias"]).tofile(f)
            f32(sd[f"{b}.linear2.weight"]).tofile(f)                 # [256,1024]
            f32(sd[f"{b}.linear2.bias"]).tofile(f)
        f32(sd[f"{rn}.linear2.weight"]).tofile(f)                    # [flat,256]
        f32(sd[f"{rn}.linear2.bias"]).tofile(f)
        # Schedules are buffers on the head, so take them rather than recomputing:
        # lerobot builds them in float32 and the engine must see the same values.
        f32(head.betas).tofile(f)
        f32(head.alphas).tofile(f)
        f32(head.alpha_hats).tofile(f)
    print(f"  head: action_dim={action_dim} horizon={cfg.chunk_size} "
          f"steps={cfg.num_diffusion_steps} max_action={cfg.max_action}")
    return action_dim


def stats_of(pipeline, key):
    """(mean, std) as recorded in the checkpoint's normalization statistics."""
    for step in getattr(pipeline, "steps", []):
        st = getattr(step, "stats", None)
        if st and key in st:
            s = st[key]
            m = s.get("mean"); d = s.get("std")
            if m is not None and d is not None:
                return f32(m).ravel(), f32(d).ravel()
    raise SystemExit(f"no mean/std for {key} in the checkpoint statistics")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="hub id or local dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t5-from", default="build/octo",
                    help="existing dump to copy the frozen t5 tower + tokenizer from")
    ap.add_argument("--task", default="pick up the tape")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    ckpt = args.checkpoint
    if not os.path.isdir(ckpt):
        from huggingface_hub import snapshot_download
        ckpt = snapshot_download(ckpt)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    print(f"Loading {ckpt} on cpu ...")
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = "cpu"
    cfg.use_amp = False
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.eval().float()
    pre, _ = make_pre_post_processors(
        cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})

    sd = policy.state_dict()
    ot = "model.octo_transformer"
    bt = f"{ot}.block_transformer.transformer"
    n_layers = 1 + max(int(k.split(".")[-3]) for k in sd
                       if k.startswith(f"{bt}.encoder_blocks.") and k.endswith("layer_norm1.weight"))
    d = int(sd[f"{bt}.layer_norm.weight"].shape[0])
    mlp = int(sd[f"{bt}.encoder_blocks.0.mlp_block.dense1.weight"].shape[0])
    n_heads = 6 if d == 384 else d // 64
    window = cfg.n_obs_steps

    print("Dumping weights ...")
    dump_stem(sd, f"{ot}.observation_tokenizers.primary.encoder", out, "stem_primary")
    dump_stem(sd, f"{ot}.observation_tokenizers.wrist.encoder", out, "stem_wrist")
    dump_transformer(sd, cfg, out, n_layers, d, n_heads, mlp, window)
    head = policy.model.heads["action"]
    action_dim = dump_head(sd, head, out, cfg)

    # frozen language tower + tokenizer, copied from the base dump
    src = os.path.expanduser(args.t5_from)
    for fn in ("t5.meta", "t5.bin"):
        if not os.path.exists(os.path.join(src, fn)):
            sys.exit(f"missing {fn} in --t5-from {src}; run convert_octo.py once")
        shutil.copy2(os.path.join(src, fn), os.path.join(out, fn))
    print(f"  t5: copied from {src} (frozen, absent from the checkpoint)")

    amean, astd = stats_of(pre, "action")
    amean.tofile(f"{out}/stats_action_mean.bin")
    astd.tofile(f"{out}/stats_action_std.bin")
    np.ones(action_dim, np.float32).tofile(f"{out}/stats_action_mask.bin")
    print(f"  stats: action {amean.shape[0]}-dim mean/std, mask all-ones")

    with open(f"{out}/config.txt", "w") as f:
        f.write(f"instruction {args.task}\nwindow {window}\n")
        f.write(f"steps {cfg.num_diffusion_steps}\n")
        f.write(f"checkpoint {args.checkpoint}\n")
        f.write(f"action_dim {action_dim}\nhorizon {cfg.chunk_size}\n")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)
                if os.path.isfile(os.path.join(out, f)))
    print(f"done -> {out} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
