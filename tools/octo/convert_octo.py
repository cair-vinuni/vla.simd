"""
Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

Convert the authoritative Octo-Small-1.5 JAX checkpoint into the flat fp32
arenas the C++ engine loads. Setup env according to README.md, then run at the
top of the repo:

  python tools/octo/convert_octo.py [OUT_DIR] [--inspect]

Checkpoint: rail-berkeley/octo-small-1.5, fetched from the HF hub on first run;
set OCTO_CKPT=/path/to/checkpoint to use a local copy instead.

Outputs (default build/octo/):
  config.txt                 text KV: all dims
  t5.meta / t5.bin           T5-base encoder (fp32: embedding, rel bias [32,12],
                             per layer ln1,q,k,v,o,ln2,wi,wo ; final ln)
  stem_primary.meta/.bin     SmallStem16 (per layer: conv w [Cout,3,3,Cin] WS-folded,
  stem_wrist.meta/.bin        conv b, gn scale, gn bias ; then embed w [512,384], b)
  octo.meta / octo.bin       projections, pos embeddings, 12 encoder blocks, final ln
  head.meta / head.bin       diffusion score net + beta schedule
  stats_action_{mean,std,mask}.bin   bridge_dataset action stats [7]
"""
import os
import sys

import numpy as np

CKPT = os.environ.get("OCTO_CKPT", "hf://rail-berkeley/octo-small-1.5")
OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "build/octo"
INSPECT = "--inspect" in sys.argv
os.makedirs(OUT, exist_ok=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OCTO_SRC = os.path.join(REPO_ROOT, "third_party", "octo")
if not os.path.isdir(OCTO_SRC):
    sys.exit(
        f"{OCTO_SRC} is missing. This script drives the authoritative JAX model, "
        "which is a separate checkout:\n"
        "  git clone https://github.com/octo-models/octo third_party/octo\n"
        "  uv pip install -e third_party/octo --no-deps\n"
        "See tools/octo/pyproject.toml for the rest of the frozen environment."
    )
sys.path.insert(0, OCTO_SRC)

# dlimp is only needed by octo's data pipeline, which we never touch; stub it out
# so we don't have to install a git dependency in the reference venv.
import types
_dl = types.ModuleType("dlimp")
_dl.DLataset = object
_dl.transforms = types.SimpleNamespace()
sys.modules.setdefault("dlimp", _dl)

import flax

from octo.model.octo_model import OctoModel

print(f"loading {CKPT} ...")
model = OctoModel.load_pretrained(CKPT)
params = model.params

if INSPECT:
    flat = flax.traverse_util.flatten_dict(params)
    for k, v in flat.items():
        print("/".join(k), v.shape, v.dtype)
    sys.exit(0)

INSTRUCTION = "pick up the black bowl"
W = 2  # window size


def f32(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


# ---- diffusion head configuration ----
head_cfg = model.config["model"]["heads"]["action"]["kwargs"]
ACT_D = head_cfg["action_dim"]            # 7
ACT_H = head_cfg["action_horizon"]        # 4
STEPS = head_cfg.get("diffusion_steps", 20)
MAXA = head_cfg.get("max_action", 5.0)

hp = params["heads_action"]


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps, dtype=np.float64) / timesteps
    ac = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return np.clip(betas, 0, 0.999)


# The schedule is data, not weights: the checkpoint stores the step count and
# the parameterization, and the engine reads the three arrays out of head.bin.
betas = cosine_beta_schedule(STEPS).astype(np.float32)
alphas = (1.0 - betas).astype(np.float32)
alpha_hats = np.cumprod(alphas).astype(np.float32)

# bridge_dataset action statistics, which the engine un-normalizes with
stats = model.dataset_statistics["bridge_dataset"]["action"]
amean, astd = f32(stats["mean"]), f32(stats["std"])
amask = f32(np.asarray(stats.get("mask", np.ones_like(amean, bool)), np.float32))
amean.tofile(f"{OUT}/stats_action_mean.bin")
astd.tofile(f"{OUT}/stats_action_std.bin")
amask.tofile(f"{OUT}/stats_action_mask.bin")

# ============================================================================
# weight export (all fp32; the checkpoint is fp32, so nothing is rounded)
# ============================================================================


def T(x):  # flax Dense kernel [in,out] -> nn.Linear [out,in]
    return f32(np.transpose(np.asarray(x)))


# ---- T5 encoder ----
t5 = params["octo_transformer"]["task_tokenizers_language"]["hf_model"]
enc = t5["encoder"]
D_M, N_L = 768, 12
rel = np.asarray(enc["block"]["0"]["layer"]["0"]["SelfAttention"]
                 ["relative_attention_bias"]["embedding"])   # [32, 12]
with open(f"{OUT}/t5.meta", "w") as f:
    f.write(f"d_model {D_M}\nn_layers {N_L}\nn_heads 12\nd_kv 64\nd_ff 3072\n")
    f.write(f"vocab {np.asarray(t5['shared']['embedding']).shape[0]}\n")
    f.write(f"n_buckets {rel.shape[0]}\nmax_dist 128\neps 1e-6\nn_tokens 16\n")
with open(f"{OUT}/t5.bin", "wb") as f:
    f32(t5["shared"]["embedding"]).tofile(f)          # [vocab, 768]
    f32(rel).tofile(f)                                # [32, 12]
    for L in range(N_L):
        blk = enc["block"][str(L)]["layer"]
        att = blk["0"]["SelfAttention"]
        f32(blk["0"]["layer_norm"]["weight"]).tofile(f)
        T(att["q"]["kernel"]).tofile(f)
        T(att["k"]["kernel"]).tofile(f)
        T(att["v"]["kernel"]).tofile(f)
        T(att["o"]["kernel"]).tofile(f)
        f32(blk["1"]["layer_norm"]["weight"]).tofile(f)
        T(blk["1"]["DenseReluDense"]["wi"]["kernel"]).tofile(f)
        T(blk["1"]["DenseReluDense"]["wo"]["kernel"]).tofile(f)
    f32(enc["final_layer_norm"]["weight"]).tofile(f)


# ---- SmallStem16 x2 ----
def weight_standardize(w, eps=1e-5):
    w = np.asarray(w, np.float64)
    m = w.mean(axis=(0, 1, 2), keepdims=True)
    s = w.std(axis=(0, 1, 2), keepdims=True)
    return ((w - m) / (s + eps)).astype(np.float32)


def dump_stem(name, key):
    stem = params["octo_transformer"][key]["SmallStem16_0"]
    feats = []
    with open(f"{OUT}/{name}.bin", "wb") as f:
        for i in range(4):
            cw = weight_standardize(stem[f"StdConv_{i}"]["kernel"])   # [3,3,Cin,Cout]
            feats.append(cw.shape[-1])
            f32(np.transpose(cw, (3, 0, 1, 2))).tofile(f)             # [Cout,3,3,Cin]
            f32(stem[f"StdConv_{i}"]["bias"]).tofile(f)
            f32(stem[f"GroupNorm_{i}"]["scale"]).tofile(f)
            f32(stem[f"GroupNorm_{i}"]["bias"]).tofile(f)
        ew = np.asarray(stem["embedding"]["kernel"])                  # [1,1,384,512]
        T(ew.reshape(ew.shape[2], ew.shape[3])).tofile(f)             # [512,384]
        f32(stem["embedding"]["bias"]).tofile(f)
    in_ch = np.asarray(stem["StdConv_0"]["kernel"]).shape[2]
    with open(f"{OUT}/{name}.meta", "w") as f:
        f.write(f"in_ch {in_ch}\nn_layers 4\nk 3\nstride 2\npad 1\n")
        f.write("features " + " ".join(map(str, feats)) + "\n")
        f.write(f"embed_dim {ew.shape[3]}\ngn_groups 32\ngn_eps 1e-6\n")


dump_stem("stem_primary", "observation_tokenizers_primary")
dump_stem("stem_wrist", "observation_tokenizers_wrist")

# ---- octo transformer ----
otp = params["octo_transformer"]
tr = otp["BlockTransformer_0"]["Transformer_0"]
D, NLYR, NH, HD, MLP = 384, 12, 6, 64, 1536
with open(f"{OUT}/octo.meta", "w") as f:
    f.write(f"d {D}\nn_layers {NLYR}\nheads {NH}\nhead_dim {HD}\nmlp {MLP}\n")
    f.write("max_horizon 10\nn_task 16\ntok_primary 256\ntok_wrist 64\nn_readout 1\n")
    f.write(f"t5_dim 768\nstem_dim 512\nln_eps 1e-6\nwindow {W}\n")
with open(f"{OUT}/octo.bin", "wb") as f:
    T(otp["task_language_projection"]["kernel"]).tofile(f)
    f32(otp["task_language_projection"]["bias"]).tofile(f)
    T(otp["obs_primary_projection"]["kernel"]).tofile(f)
    f32(otp["obs_primary_projection"]["bias"]).tofile(f)
    T(otp["obs_wrist_projection"]["kernel"]).tofile(f)
    f32(otp["obs_wrist_projection"]["bias"]).tofile(f)
    f32(otp["task_language_pos_embedding"]).tofile(f)      # [1,16,384]
    f32(otp["obs_primary_pos_embedding"]).tofile(f)        # [1,10,256,384]
    f32(otp["obs_wrist_pos_embedding"]).tofile(f)          # [1,10,64,384]
    f32(otp["readout_action_pos_embedding"]).tofile(f)     # [1,10,1,384]
    for L in range(NLYR):
        blk = tr[f"encoderblock_{L}"]
        att = blk["MultiHeadDotProductAttention_0"]
        f32(blk["LayerNorm_0"]["scale"]).tofile(f)
        f32(blk["LayerNorm_0"]["bias"]).tofile(f)
        for nm in ("query", "key", "value"):
            T(np.asarray(att[nm]["kernel"]).reshape(D, NH * HD)).tofile(f)  # [384,384] out,in
            f32(att[nm]["bias"]).tofile(f)                                  # [6,64] -> flat 384
        T(np.asarray(att["out"]["kernel"]).reshape(NH * HD, D)).tofile(f)   # [384,384] out,in
        f32(att["out"]["bias"]).tofile(f)
        f32(blk["LayerNorm_1"]["scale"]).tofile(f)
        f32(blk["LayerNorm_1"]["bias"]).tofile(f)
        f32(T(blk["MlpBlock_0"]["Dense_0"]["kernel"])).tofile(f)
        f32(blk["MlpBlock_0"]["Dense_0"]["bias"]).tofile(f)
        f32(T(blk["MlpBlock_0"]["Dense_1"]["kernel"])).tofile(f)
        f32(blk["MlpBlock_0"]["Dense_1"]["bias"]).tofile(f)
    f32(tr["encoder_norm"]["scale"]).tofile(f)
    f32(tr["encoder_norm"]["bias"]).tofile(f)

# ---- diffusion head ----
dm = hp["diffusion_model"]
rn = dm["reverse_network"]
with open(f"{OUT}/head.meta", "w") as f:
    f.write(f"emb {D}\naction_dim {ACT_D}\nhorizon {ACT_H}\ntime_dim 32\n")
    f.write(f"num_blocks 3\nhidden 256\nsteps {STEPS}\nmax_action {MAXA}\n")
with open(f"{OUT}/head.bin", "wb") as f:
    f32(dm["time_preprocess"]["kernel"]).tofile(f)         # [16,1] -> 16
    T(dm["cond_encoder"]["Dense_0"]["kernel"]).tofile(f)   # [64,32]
    f32(dm["cond_encoder"]["Dense_0"]["bias"]).tofile(f)
    T(dm["cond_encoder"]["Dense_1"]["kernel"]).tofile(f)   # [32,64]
    f32(dm["cond_encoder"]["Dense_1"]["bias"]).tofile(f)
    T(rn["Dense_0"]["kernel"]).tofile(f)                   # [256,444]
    f32(rn["Dense_0"]["bias"]).tofile(f)
    for i in range(3):
        b = rn[f"MLPResNetBlock_{i}"]
        f32(b["LayerNorm_0"]["scale"]).tofile(f)
        f32(b["LayerNorm_0"]["bias"]).tofile(f)
        T(b["Dense_0"]["kernel"]).tofile(f)                # [1024,256]
        f32(b["Dense_0"]["bias"]).tofile(f)
        T(b["Dense_1"]["kernel"]).tofile(f)                # [256,1024]
        f32(b["Dense_1"]["bias"]).tofile(f)
    T(rn["Dense_1"]["kernel"]).tofile(f)                   # [28,256]
    f32(rn["Dense_1"]["bias"]).tofile(f)
    betas.tofile(f)
    alphas.tofile(f)
    alpha_hats.tofile(f)

with open(f"{OUT}/config.txt", "w") as f:
    f.write(f"instruction {INSTRUCTION}\nwindow {W}\nsteps {STEPS}\n")
    f.write("dataset bridge_dataset\n")

print(f"done -> {OUT}")
