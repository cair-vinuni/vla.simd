#!/usr/bin/env python3
"""
convert_hf_safetensors.py

Convert a lerobot SmolVLA checkpoint into the vla.simd .meta/.bin arena **without
torch, lerobot or transformers** - it reads `model.safetensors` and the processor
state files directly. Byte layouts are identical to
`convert_lerobot_ckpt.py`, which is the torch-based converter and remains the
reference; this one exists so a serving box (a Raspberry Pi, here) can convert a
checkpoint it could never load through the training stack.

    python3 tools/convert_hf_safetensors.py khanhnd61/smolvla_so101_tape_prune6 \
        build/smolvla --task "pick up the tape" --pos-ids identity

Accepts a Hub id (downloaded to ~/.cache/vla_simd) or a local directory holding
config.json / model.safetensors / policy_*.json / the two normalizer
safetensors. The SmolVLM2 tokenizer named by the preprocessor is fetched the same
way and flattened into <out>/tok/.

What it must reproduce, and cannot read off the checkpoint:

  * **Layer count.** `num_vlm_layers` in config.json is the count *before*
    `prune_vlm_layers` is applied; a pruned checkpoint stores fewer. The layer
    count comes from the tensor names in the safetensors header, never from the
    config.
  * **SigLIP position ids.** SmolVLM buckets fractional patch coordinates rather
    than indexing 0..n-1, and transformers 4.55-4.57 shifted every bucket down by
    one (`k/n*(1-1e-6)` replaced `arange(0, 1-1e-6, 1/n)`); 5.0 restored the
    identity. The torch converter captures the mapping from the installed
    transformers with a forward hook; with no transformers to ask, `--pos-ids`
    must be given. Getting this wrong loads, runs, and is quietly wrong - the ViT
    alone drops to cos=0.86.

Everything else is structural and is asserted against the header rather than
assumed: dtypes, shapes, and which expert layers are cross-attention.
"""

import argparse
import json
import mmap
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROPE_BASE = 10000.0   # apply_rope's hardcoded default, NOT text_config.rope_theta
HUB = "https://huggingface.co/{repo}/resolve/main/{path}"
CACHE = Path(os.environ.get("VLA_SIMD_CACHE", "~/.cache/vla_simd")).expanduser()

CKPT_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json",
              "policy_postprocessor.json")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def fetch(repo, path, kind="models"):
    """Return a local path for <repo>/<path>, downloading into CACHE if needed."""
    dst = CACHE / repo.replace("/", "--") / path
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = HUB.format(repo=repo, path=path)
    if kind != "models":
        url = url.replace("huggingface.co/", f"huggingface.co/{kind}/")
    log(f"  fetching {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def resolve(src):
    """Local checkpoint dir, or a Hub id whose files are pulled into the cache."""
    p = Path(src).expanduser()
    if p.is_dir():
        if not (p / "model.safetensors").is_file():
            sys.exit(f"{p} has no model.safetensors")
        return p
    for f in CKPT_FILES:
        fetch(src, f)
    d = CACHE / src.replace("/", "--")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in json.loads((d / name).read_text())["steps"]:
            if step.get("state_file"):
                fetch(src, step["state_file"])
    return d


# ---------------------------------------------------------------------------
# safetensors, read directly (no `safetensors` package)
# ---------------------------------------------------------------------------
DTYPES = {"F32": np.float32, "F64": np.float64, "F16": np.float16,
          "I64": np.int64, "I32": np.int32, "BF16": np.uint16, "BOOL": np.bool_}


class Safetensors:
    """mmap'd reader. BF16 tensors come back as raw uint16 (`bits`) or widened."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.meta = self.header.pop("__metadata__", {})

    def __contains__(self, k):
        return k in self.header

    def keys(self):
        return [k for k in self.header]

    def dtype(self, k):
        return self.header[k]["dtype"]

    def shape(self, k):
        return tuple(self.header[k]["shape"])

    def bits(self, k):
        """Raw tensor as its stored dtype (BF16 -> uint16), shape preserved."""
        h = self.header[k]
        a, b = h["data_offsets"]
        arr = np.frombuffer(self.mm, dtype=DTYPES[h["dtype"]],
                            count=(b - a) // np.dtype(DTYPES[h["dtype"]]).itemsize,
                            offset=self.base + a)
        return arr.reshape(h["shape"])

    def f32(self, k):
        """Tensor as fp32. Widening bf16 is exact (it is a truncated fp32)."""
        raw = self.bits(k)
        if self.dtype(k) == "BF16":
            out = np.zeros(raw.shape + (2,), np.uint16)
            out[..., 1] = raw                       # bf16 occupies the high half
            return out.view(np.float32).reshape(raw.shape)
        return np.ascontiguousarray(raw, np.float32)

    def bf16(self, k):
        """Tensor as bf16 bits. Stored bf16 passes through untouched."""
        if self.dtype(k) == "BF16":
            return np.ascontiguousarray(self.bits(k))
        return f32_to_bf16(self.f32(k))


def f32_to_bf16(x):
    """Round-to-nearest-even fp32 -> bf16, matching torch's .to(bfloat16)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# SigLIP position ids
# ---------------------------------------------------------------------------
def position_ids(side, mode):
    """SmolVLMVisionEmbeddings' patch -> position-row mapping for a full image.

    boundaries = arange(1/n, 1, 1/n); the coordinate of patch k is bucketized with
    torch.bucketize(..., right=True), i.e. the first boundary strictly greater.
      identity (<=4.54, >=5.0): coord = arange(0, 1-1e-6, 1/n)  -> [0, 1, ..., n-1]
      shifted  (4.55-4.57):     coord = k/n * (1-1e-6)          -> [0, 0, 1, ..., n-2]
    """
    boundaries = np.arange(1, side, dtype=np.float64) / side
    k = np.arange(side, dtype=np.float64)
    coord = (k / side) * (1 - 1e-6) if mode == "shifted" else k / side
    bucket = np.searchsorted(boundaries, coord.astype(np.float32).astype(np.float64),
                             side="right")
    return (bucket[:, None] * side + bucket[None, :]).reshape(-1).astype(np.int64)


# ---------------------------------------------------------------------------
# weight export - byte layouts must match the engine's loaders
# (src/models/smolvla/{smollm2_lm,siglip_vision,action_expert,smolvla_model}.cpp)
# ---------------------------------------------------------------------------
VLM = "model.vlm_with_expert.vlm.model.text_model"
VIT = "model.vlm_with_expert.vlm.model.vision_model"
AEX = "model.vlm_with_expert.lm_expert"


def n_layers_of(st, prefix):
    """Layer count from the tensor names. config.json's num_vlm_layers is pre-prune."""
    idx = set()
    for k in st.keys():
        if k.startswith(prefix + ".layers."):
            idx.add(int(k[len(prefix) + 8:].split(".")[0]))
    if idx != set(range(len(idx))):
        sys.exit(f"{prefix}: non-contiguous layer indices {sorted(idx)}")
    return len(idx)


def dump_vlm(st, out, cfg_text, n_layers):
    with open(f"{out}/vlm.meta", "w") as f:
        f.write(f"hidden {cfg_text['hidden']}\nn_q {cfg_text['n_q']}\n")
        f.write(f"n_kv {cfg_text['n_kv']}\nhead_dim {cfg_text['head_dim']}\n")
        f.write(f"ffn {cfg_text['ffn']}\n")
        f.write(f"eps {cfg_text['eps']}\nrope_base {ROPE_BASE}\nn_layers {n_layers}\n")
    with open(f"{out}/vlm.bin", "wb") as f:
        for L in range(n_layers):                   # fp32 region: per-layer norms
            st.f32(f"{VLM}.layers.{L}.input_layernorm.weight").tofile(f)
            st.f32(f"{VLM}.layers.{L}.post_attention_layernorm.weight").tofile(f)
        st.f32(f"{VLM}.norm.weight").tofile(f)      # output norm
        for L in range(n_layers):                   # bf16 region: matmul weights
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                st.bf16(f"{VLM}.layers.{L}.{name}.weight").tofile(f)


def dump_vit(st, out, vc, n_img_tok, mm_out, pos_mode):
    nl, patch, img = vc["n_layers"], vc["patch"], vc["img"]
    side = img // patch
    n_patches = side * side
    pos = position_ids(side, pos_mode)
    log(f"  siglip position ids: {pos[:4].tolist()} ... {pos[-2:].tolist()} ({pos_mode})")
    with open(f"{out}/vit.meta", "w") as f:
        f.write(f"hidden {vc['hidden']}\nn_heads {vc['n_heads']}\n")
        f.write(f"head_dim {vc['hidden'] // vc['n_heads']}\ninter {vc['inter']}\n")
        f.write(f"n_layers {nl}\npatch {patch}\nimg {img}\nn_patches {n_patches}\n")
        f.write(f"ln_eps {vc['ln_eps']}\nscale_factor {vc['scale_factor']}\n")
        f.write(f"mm_out {mm_out}\nn_img_tok {n_img_tok}\n")
    with open(f"{out}/vit.bin", "wb") as f:
        st.f32(f"{VIT}.embeddings.patch_embedding.bias").tofile(f)
        # gathered to patch order: the engine adds position row p to patch p
        st.f32(f"{VIT}.embeddings.position_embedding.weight")[pos].tofile(f)
        for L in range(nl):
            p = f"{VIT}.encoder.layers.{L}"
            for name in ("layer_norm1.weight", "layer_norm1.bias",
                         "self_attn.q_proj.bias", "self_attn.k_proj.bias",
                         "self_attn.v_proj.bias", "self_attn.out_proj.bias",
                         "layer_norm2.weight", "layer_norm2.bias",
                         "mlp.fc1.bias", "mlp.fc2.bias"):
                st.f32(f"{p}.{name}").tofile(f)
        st.f32(f"{VIT}.post_layernorm.weight").tofile(f)
        st.f32(f"{VIT}.post_layernorm.bias").tofile(f)
        # [768,3,16,16] C-order is exactly the [768, 3*16*16] the engine GEMMs against
        st.bf16(f"{VIT}.embeddings.patch_embedding.weight").tofile(f)
        for L in range(nl):
            p = f"{VIT}.encoder.layers.{L}"
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.out_proj", "mlp.fc1", "mlp.fc2"):
                st.bf16(f"{p}.{name}.weight").tofile(f)
        st.bf16("model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight").tofile(f)


def dump_aex(st, out, cfg, cfg_text, n_layers, expert_h, expert_ffn, san):
    with open(f"{out}/aex.meta", "w") as f:
        f.write(f"expert_h {expert_h}\nexpert_ffn {expert_ffn}\nn_q {cfg_text['n_q']}\n")
        f.write(f"n_kv {cfg_text['n_kv']}\nhead_dim {cfg_text['head_dim']}\n")
        f.write(f"eps {cfg_text['eps']}\nrope_base {ROPE_BASE}\nn_layers {n_layers}\n")
        f.write(f"self_attn_every_n {san}\n")
        f.write(f"chunk {cfg['chunk_size']}\nnum_steps {cfg['num_steps']}\n")
        f.write(f"max_action_dim {cfg['max_action_dim']}\n")
        f.write(f"min_period {cfg['min_period']}\nmax_period {cfg['max_period']}\n")
    with open(f"{out}/aex.bin", "wb") as f:
        for L in range(n_layers):
            p = f"{AEX}.layers.{L}"
            for name in ("input_layernorm.weight",
                         "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                         "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                         "post_attention_layernorm.weight",
                         "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                st.f32(f"{p}.{name}").tofile(f)
        st.f32(f"{AEX}.norm.weight").tofile(f)
        for name in ("action_in_proj", "action_time_mlp_in", "action_time_mlp_out",
                     "action_out_proj"):
            st.f32(f"model.{name}.weight").tofile(f)
            st.f32(f"model.{name}.bias").tofile(f)


def check_expert_attn(st, n_layers, san, expert_h, kv_full):
    """Even layers project k/v from the expert stream, odd ones reproject the VLM cache."""
    for L in range(n_layers):
        want = (kv_full, expert_h) if L % san == 0 else (kv_full, kv_full)
        got = st.shape(f"{AEX}.layers.{L}.self_attn.k_proj.weight")
        if got != want:
            sys.exit(f"expert layer {L}: k_proj {got}, expected {want} for "
                     f"self_attn_every_n={san}. The engine's self/cross split would be wrong.")


def dump_tokenizer(tok_json, out):
    """Flat vocab/merges for the engine's byte-level BPE tokenizer."""
    tj = json.loads(Path(tok_json).read_text(encoding="utf-8"))
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/vocab.txt", "w", encoding="utf-8") as f:
        for tok, i in vocab.items():
            f.write(f"{i}\t{tok}\n")
    with open(f"{out}/merges.txt", "w", encoding="utf-8") as f:
        for m in merges:
            f.write((m if isinstance(m, str) else f"{m[0]} {m[1]}") + "\n")
    return len(vocab), len(merges), tj


def pad_token_id(tj, cfg_pad="<|im_end|>"):
    for a in tj.get("added_tokens", []):
        if a["content"] == cfg_pad:
            return a["id"]
    return tj["model"]["vocab"].get(cfg_pad, 2)


# ---------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Hub id or local dir with model.safetensors")
    p.add_argument("out", help="Output dir for the .meta/.bin weights")
    p.add_argument("--task", default=None,
                   help="Instruction written to config.txt (the robot's default prompt)")
    p.add_argument("--pos-ids", choices=("shifted", "identity"), required=True,
                   help="SigLIP position-id mapping of the transformers the checkpoint was "
                        "trained with: shifted = 4.55-4.57.x, identity = <=4.54 or >=5.0 "
                        "(lerobot >=0.5 trains with transformers 5.x -> identity)")
    p.add_argument("--tokenizer", default=None,
                   help="Override the tokenizer repo id named by the preprocessor")
    args = p.parse_args()

    ckpt = resolve(args.checkpoint)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    log(f"Reading {ckpt}")

    cfg = json.loads((ckpt / "config.json").read_text())
    for k in ("add_image_special_tokens", "adapt_to_pi_aloha"):
        if cfg.get(k):
            sys.exit(f"{k}=True is not implemented by the engine")
    pre = json.loads((ckpt / "policy_preprocessor.json").read_text())
    post = json.loads((ckpt / "policy_postprocessor.json").read_text())
    st = Safetensors(ckpt / "model.safetensors")

    # ---- dims, from the tensors rather than the config ----
    n_layers = n_layers_of(st, VLM)
    n_expert = n_layers_of(st, AEX)
    if n_expert != n_layers:
        sys.exit(f"vlm has {n_layers} layers, expert has {n_expert}")
    hidden = st.shape(f"{VLM}.layers.0.self_attn.q_proj.weight")[0]
    kv_full = st.shape(f"{VLM}.layers.0.self_attn.k_proj.weight")[0]
    head_dim = 64                                    # SmolLM2: hidden/n_q, fixed at 64
    cfg_text = {"hidden": hidden, "n_q": hidden // head_dim, "n_kv": kv_full // head_dim,
                "head_dim": head_dim, "ffn": st.shape(f"{VLM}.layers.0.mlp.gate_proj.weight")[0],
                "eps": 1e-5}
    expert_h = st.shape(f"{AEX}.layers.0.self_attn.q_proj.weight")[1]
    expert_ffn = st.shape(f"{AEX}.layers.0.mlp.gate_proj.weight")[0]
    san = cfg["self_attn_every_n_layers"] if "cross" in cfg.get("attention_mode", "cross_attn") else 1
    if san <= 0:
        sys.exit(f"attention_mode={cfg.get('attention_mode')} with self_attn_every_n_layers={san} "
                 "(all-cross expert) is not supported")
    check_expert_attn(st, n_layers, san, expert_h, kv_full)

    vh = st.shape(f"{VIT}.encoder.layers.0.self_attn.q_proj.weight")[0]
    patch = st.shape(f"{VIT}.embeddings.patch_embedding.weight")[2]
    img = cfg["resize_imgs_with_padding"][0]
    n_pos = st.shape(f"{VIT}.embeddings.position_embedding.weight")[0]
    if n_pos != (img // patch) ** 2:
        sys.exit(f"position table {n_pos} != {(img // patch) ** 2} patches at {img}/{patch}")
    sf = st.shape("model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight")[1]
    sf = int(round((sf / vh) ** 0.5))                # shuffled_dim = hidden * sf^2
    vit_heads = {768: 12, 1152: 16}.get(vh) or sys.exit(
        f"unknown SigLIP width {vh}: head count not derivable")
    vc = {"hidden": vh, "n_heads": vit_heads, "inter": st.shape(f"{VIT}.encoder.layers.0.mlp.fc1.weight")[0],
          "n_layers": n_layers_of(st, f"{VIT}.encoder"), "patch": patch, "img": img,
          "ln_eps": 1e-6, "scale_factor": sf}
    n_img_tok = n_pos // (sf * sf)

    # cameras actually fed: the rename map is what the robot's keys become
    rename = next((s["config"]["rename_map"] for s in pre["steps"]
                   if s["registry_name"] == "rename_observations_processor"), {})
    cams = [k for k, v in cfg["input_features"].items()
            if v["type"] == "VISUAL" and (not rename or k in rename.values())]
    n_views = len(cams)

    log(f"  vlm layers={n_layers} (config says {cfg['num_vlm_layers']} pre-prune"
        f"{', pruned ' + str(cfg['prune_vlm_layers']) if cfg.get('prune_vlm_layers') else ''})"
        f" hidden={hidden} expert_h={expert_h} ffn={expert_ffn} self_attn_every_n={san}")
    log(f"  vit layers={vc['n_layers']} hidden={vh} patch={patch} img={img} "
        f"scale_factor={sf} -> {n_img_tok} tokens/view")
    log(f"  views={n_views} {[c.split('.')[-1] for c in cams]} "
        f"(from {[k.split('.')[-1] for k in rename]})")

    log("Dumping weights ...")
    dump_vlm(st, out, cfg_text, n_layers)
    dump_vit(st, out, vc, n_img_tok, hidden, args.pos_ids)
    dump_aex(st, out, cfg, cfg_text, n_layers, expert_h, expert_ffn, san)
    st.bf16(f"{VLM}.embed_tokens.weight").tofile(f"{out}/emb.bin")
    with open(f"{out}/heads.bin", "wb") as f:
        st.f32("model.state_proj.weight").tofile(f)
        st.f32("model.state_proj.bias").tofile(f)

    # ---- normalization statistics ----
    def stats(spec, base, key, ftype):
        step = next(s for s in spec["steps"] if s.get("state_file"))
        mode = step["config"].get("norm_map", {}).get(ftype, "IDENTITY")
        if mode != "MEAN_STD":
            sys.exit(f"{key} is normalized {mode}; the engine implements MEAN_STD only")
        n = Safetensors(base / step["state_file"])
        return n.f32(f"{key}.mean").reshape(-1), n.f32(f"{key}.std").reshape(-1)

    smean, sstd = stats(pre, ckpt, "observation.state", "STATE")
    amean, astd = stats(post, ckpt, "action", "ACTION")
    smean.tofile(f"{out}/stats_state_mean.bin"); sstd.tofile(f"{out}/stats_state_std.bin")
    amean.tofile(f"{out}/stats_action_mean.bin"); astd.tofile(f"{out}/stats_action_std.bin")
    log(f"  state stats {smean.shape[0]}-dim, action stats {amean.shape[0]}-dim")

    with open(f"{out}/heads.meta", "w") as f:
        f.write(f"vocab {st.shape(f'{VLM}.embed_tokens.weight')[0]}\nhidden {hidden}\n")
        f.write(f"max_state_dim {cfg['max_state_dim']}\n")
        f.write(f"real_state_dim {smean.shape[0]}\nreal_action_dim {amean.shape[0]}\n")
        f.write(f"n_views {n_views}\n")

    # ---- tokenizer ----
    tok_step = next(s for s in pre["steps"] if s["registry_name"] == "tokenizer_processor")
    tok_repo = args.tokenizer or tok_step["config"]["tokenizer_name"]
    tok_path = Path(tok_repo).expanduser() / "tokenizer.json" if Path(tok_repo).is_dir() \
        else fetch(tok_repo, "tokenizer.json")
    nv, nm, tj = dump_tokenizer(tok_path, f"{out}/tok")
    log(f"  tokenizer {tok_repo}: vocab={nv} merges={nm}")

    task = args.task or "do the task"
    with open(f"{out}/config.txt", "w") as f:
        f.write(f"instruction {task}\n")
        f.write(f"tokenizer_max_length {tok_step['config']['max_length']}\n")
        f.write(f"pad_token_id {pad_token_id(tj)}\n")
        f.write(f"chunk {cfg['chunk_size']}\nnum_steps {cfg['num_steps']}\nn_views {n_views}\n")
        f.write(f"checkpoint {args.checkpoint}\n")
        f.write(f"pos_ids {args.pos_ids}\n")
        for i, c in enumerate(cams):
            f.write(f"cam{i} {c.split('.')[-1]}\n")
        for src, dst in rename.items():
            f.write(f"rename {src.split('.')[-1]} {dst.split('.')[-1]}\n")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)
                if os.path.isfile(os.path.join(out, f)))
    log(f"done -> {out} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
