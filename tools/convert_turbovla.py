#!/usr/bin/env python3
"""
convert_turbovla.py -- convert a TurboVLA LIBERO checkpoint into the flat
.meta/.bin arenas the vla.simd engine loads.

    python tools/convert_turbovla.py \
        --ckpt build/turbovla_ckpt/object.pth \
        --out  build/turbovla_object

Run it in a venv with torch + transformers>=4.57 (see docs/12-turbovla-design.md).
The engine itself needs none of that at runtime.

The reference model is the upstream code in third_party/TurboVLA, driven with the
real checkpoint weights. Two substitutions make that possible without HF access:

  * DINOv3 is a GATED repo, but its *weights* are inside the TurboVLA checkpoint
    (the vision tower is fine-tuned, not frozen). Only the architecture config is
    needed, and that is public -- assets/dinov3_vitb16_config.json is a copy of
    the config from the ungated onnx-community mirror. So AutoModel.from_pretrained
    is patched to build DINOv3ViTModel from that config; the random init it
    returns is then overwritten by the checkpoint's strict load_state_dict.
  * BERT gets the same treatment (config only); its tokenizer files are a plain
    ungated download.

Written into <out>/:
    config.txt    dims, epsilons, image normalization, special token ids
    stats.bin     proprio mean/std, action min/max (LIBERO eval protocol)
    vocab.txt     BERT WordPiece vocabulary, one token per line
    text_pad.txt  instruction -> padded text length (see the note below)
    vision.meta   DINOv3 ViT-B/16 dims
    vision.bin    patch embed + 12 blocks, LayerScale folded into o_proj/down_proj
    text.meta     BERT dims
    text.bin      embeddings + 12 blocks + the 768->256 text projection
    fusion.meta   interaction dims
    fusion.bin    vision projection, view embeddings, 6x (bi-attention + text layer)
    head.meta     action head dims
    head.bin      state projection, 3 decoder layers, action MLP

text_pad.txt is not an optimization. The action decoder attends the *padded*
text tokens as well (encode_condition concatenates them with no mask), so the
padded length is part of the model's output, and the checkpoint pins a length
per training instruction (11/14/21 for the LIBERO suites). Rows past the group
length are literal zeros in BERT's output, which the text projection turns into
its own bias -- reproduced here so the engine can reproduce it too.
"""

import argparse
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "third_party", "TurboVLA")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# reference model
# ---------------------------------------------------------------------------
def stub_timm():
    """turbovla.models.components.fusion imports timm only for DropPath, which is
    the identity outside training. Stubbing it keeps the venv to torch+transformers
    (timm pulls torchvision, which pins a torch build we do not want here)."""
    if "timm" in sys.modules:
        return
    import torch.nn as nn

    layers = types.ModuleType("timm.models.layers")

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            raise RuntimeError("DropPath stub is inference-only")

    layers.DropPath = DropPath
    models = types.ModuleType("timm.models")
    models.layers = layers
    timm = types.ModuleType("timm")
    timm.models = models
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers


def build_reference(ckpt_path, bert_dir, dinov3_config_path):
    """Load the checkpoint into the upstream TurboVLA module tree (strict)."""
    import torch
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
    from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel

    if not os.path.isdir(REF):
        sys.exit(f"reference code not found at {REF}\n"
                 "  git clone https://github.com/H-EmbodVis/TurboVLA third_party/TurboVLA")
    stub_timm()
    sys.path.insert(0, REF)

    with open(dinov3_config_path) as f:
        dino_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("transformers")}
    dino_cfg.pop("architectures", None)
    dino_cfg.pop("torch_dtype", None)
    bert_cfg = BertConfig.from_pretrained(bert_dir)

    real_from_pretrained = AutoModel.from_pretrained

    def from_config_only(name_or_path, **kwargs):
        name = str(name_or_path).lower()
        if "dinov3" in name:
            return DINOv3ViTModel(DINOv3ViTConfig(**dino_cfg))
        if "bert" in name:
            return BertModel(bert_cfg)
        return real_from_pretrained(name_or_path, **kwargs)

    AutoModel.from_pretrained = from_config_only
    real_tokenizer = AutoTokenizer.from_pretrained

    def tokenizer_from_local(name_or_path, **kwargs):
        if "bert" in str(name_or_path).lower():
            kwargs.pop("local_files_only", None)
            return real_tokenizer(bert_dir, **kwargs)
        return real_tokenizer(name_or_path, **kwargs)

    AutoTokenizer.from_pretrained = tokenizer_from_local
    try:
        from turbovla.models.configuration import TurboVLAConfig
        from turbovla.models.turbovla import build_turbovla

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_config" not in checkpoint:
            sys.exit(f"{ckpt_path} has no model_config -- not a released TurboVLA checkpoint")
        config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
        model = build_turbovla(config)
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
    finally:
        AutoModel.from_pretrained = real_from_pretrained
        AutoTokenizer.from_pretrained = real_tokenizer

    model.eval().float()
    model.requires_grad_(False)
    log(f"reference: {checkpoint.get('model_name')} suite={checkpoint.get('suite')} "
        f"step={checkpoint.get('global_step')} ({len(state)} tensors)")
    return model, config, checkpoint


# ---------------------------------------------------------------------------
# weight arena helpers -- mirror the take() cursor in the C++ loaders
# ---------------------------------------------------------------------------
class Arena:
    """Append-only float32 blob."""

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
    import torch

    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


def linear(arena, mod, shape=None):
    """nn.Linear -> W [N, K] then bias [N] (the layout nn::Linear::init expects)."""
    arena.add(mod.weight, shape)
    if mod.bias is None:
        raise AssertionError("bias-less Linear: the loader must skip it explicitly")
    arena.add(mod.bias, (mod.weight.shape[0],))


def layernorm(arena, mod):
    arena.add(mod.weight).add(mod.bias)


def mha(arena, mod, dim):
    """nn.MultiheadAttention packs q,k,v in one in_proj_weight [3D, D]; the engine
    keeps four separate nn::Linear, so split it here once."""
    w, b = mod.in_proj_weight, mod.in_proj_bias
    assert tuple(w.shape) == (3 * dim, dim), tuple(w.shape)
    for i in range(3):
        arena.add(w[i * dim:(i + 1) * dim]).add(b[i * dim:(i + 1) * dim])
    linear(arena, mod.out_proj, (dim, dim))


# ---------------------------------------------------------------------------
# module dumps
# ---------------------------------------------------------------------------
def dump_vision(model, out_dir):
    """DINOv3 ViT-B/16.

    Two liberties, both exact-in-intent and documented in the design notes:

      * LayerScale is folded into the projection in front of it. lambda1 is a
        per-channel scale on the block output, and o_proj / down_proj write
        exactly those channels, so scaling their weight rows and biases is the
        same function (one extra fp32 rounding per weight). What is left is
        precisely nn::EncoderLayer's shape: LN -> attn -> residual -> LN -> MLP
        -> residual.
      * backbone.norm is NOT exported. The TurboVLA vision encoder reads
        outputs.hidden_states[-1], which transformers records at the last
        DINOv3ViTLayer -- before the final LayerNorm. Those weights are dead in
        this model.
    """
    backbone = model.vision_encoder.backbone
    cfg = backbone.config
    arena = Arena()
    H, I = cfg.hidden_size, cfg.intermediate_size

    emb = backbone.embeddings
    arena.add(emb.cls_token, (1, 1, H))
    arena.add(emb.register_tokens, (1, cfg.num_register_tokens, H))
    # torch conv weight [Cout, Cin, k, k] -> the engine's patch-major [Cout, Cin*k*k],
    # laid out (ic, kh, kw) to match extract_patches() in the engine.
    pw = emb.patch_embeddings.weight
    arena.add(pw.reshape(H, -1), (H, 3 * cfg.patch_size * cfg.patch_size))
    arena.add(emb.patch_embeddings.bias, (H,))

    for layer in backbone.layer:
        lam1 = layer.layer_scale1.lambda1
        lam2 = layer.layer_scale2.lambda1
        layernorm(arena, layer.norm1)
        att = layer.attention
        linear(arena, att.q_proj, (H, H))
        arena.add(att.k_proj.weight, (H, H))          # key_bias=false: no bias tensor
        linear(arena, att.v_proj, (H, H))
        arena.add(att.o_proj.weight * lam1[:, None], (H, H))
        arena.add(att.o_proj.bias * lam1, (H,))
        layernorm(arena, layer.norm2)
        linear(arena, layer.mlp.up_proj, (I, H))
        arena.add(layer.mlp.down_proj.weight * lam2[:, None], (H, I))
        arena.add(layer.mlp.down_proj.bias * lam2, (H,))

    meta = [
        f"hidden {H}",
        f"n_heads {cfg.num_attention_heads}",
        f"head_dim {H // cfg.num_attention_heads}",
        f"inter {I}",
        f"n_layers {cfg.num_hidden_layers}",
        f"patch {cfg.patch_size}",
        f"prefix {cfg.num_register_tokens + 1}",
        f"rope_theta {float(cfg.rope_theta):g}",
        f"ln_eps {float(cfg.layer_norm_eps):g}",
    ]
    write_meta(out_dir, "vision", meta)
    n = arena.write(os.path.join(out_dir, "vision.bin"))
    log(f"  vision.bin {n * 4 / 1e6:.1f} MB")


def dump_text(model, out_dir):
    """BERT-base encoder + the 768 -> 256 text projection."""
    bert = model.text_encoder.bert
    cfg = bert.config
    arena = Arena()
    H, I = cfg.hidden_size, cfg.intermediate_size

    emb = bert.embeddings
    arena.add(emb.word_embeddings.weight, (cfg.vocab_size, H))
    arena.add(emb.position_embeddings.weight, (cfg.max_position_embeddings, H))
    arena.add(emb.token_type_embeddings.weight, (cfg.type_vocab_size, H))
    layernorm(arena, emb.LayerNorm)

    for layer in bert.encoder.layer:
        linear(arena, layer.attention.self.query, (H, H))
        linear(arena, layer.attention.self.key, (H, H))
        linear(arena, layer.attention.self.value, (H, H))
        linear(arena, layer.attention.output.dense, (H, H))
        layernorm(arena, layer.attention.output.LayerNorm)
        linear(arena, layer.intermediate.dense, (I, H))
        linear(arena, layer.output.dense, (H, I))
        layernorm(arena, layer.output.LayerNorm)

    # bert.pooler is in the checkpoint but never runs: TurboVLA reads
    # last_hidden_state, not pooler_output.
    linear(arena, model.text_encoder.text_projection, (model.config.interaction.hidden_dim, H))

    meta = [
        f"hidden {H}",
        f"n_heads {cfg.num_attention_heads}",
        f"head_dim {H // cfg.num_attention_heads}",
        f"inter {I}",
        f"n_layers {cfg.num_hidden_layers}",
        f"vocab {cfg.vocab_size}",
        f"max_pos {cfg.max_position_embeddings}",
        f"type_vocab {cfg.type_vocab_size}",
        f"ln_eps {float(cfg.layer_norm_eps):g}",
    ]
    write_meta(out_dir, "text", meta)
    n = arena.write(os.path.join(out_dir, "text.bin"))
    log(f"  text.bin {n * 4 / 1e6:.1f} MB")


def dump_fusion(model, out_dir):
    """Vision projection + view embeddings + the 6 interaction steps."""
    icfg = model.config.interaction
    D = icfg.hidden_dim
    E = icfg.enhancer_inner_dim
    vis_dim = model.vision_encoder.hidden_size
    arena = Arena()

    proj = model.vision_projection
    layernorm(arena, proj.input_norm)
    linear(arena, proj.mlp[0], (E, vis_dim))
    linear(arena, proj.mlp[3], (D, E))
    arena.add(proj.skip.weight, (D, vis_dim))          # skip has no bias
    layernorm(arena, proj.output_norm)
    arena.add(model.view_embedding, (1, model.num_views, D))

    interaction = model.vision_language_interaction
    # strict: a mismatched pair would silently dump a short arena the C++ loader
    # then reads past. The two lists are built together upstream, so this holds.
    for fusion, text in zip(interaction.fusion_layers, interaction.text_layers, strict=True):
        layernorm(arena, fusion.layer_norm_v)
        layernorm(arena, fusion.layer_norm_l)
        linear(arena, fusion.attn.v_proj, (E, D))
        linear(arena, fusion.attn.l_proj, (E, D))
        linear(arena, fusion.attn.values_v_proj, (E, D))
        linear(arena, fusion.attn.values_l_proj, (E, D))
        linear(arena, fusion.attn.out_v_proj, (D, E))
        linear(arena, fusion.attn.out_l_proj, (D, E))
        arena.add(fusion.gamma_v, (D,))
        arena.add(fusion.gamma_l, (D,))

        mha(arena, text.self_attn, D)
        layernorm(arena, text.norm1)
        linear(arena, text.linear1, (icfg.enhancer_inner_dim, D))
        linear(arena, text.linear2, (D, icfg.enhancer_inner_dim))
        layernorm(arena, text.norm2)

    meta = [
        f"hidden {D}",
        f"embed {E}",
        f"n_layers {icfg.num_layers}",
        f"fusion_heads {max(1, icfg.nheads // 2)}",
        f"text_heads {max(1, icfg.nheads // 2)}",
        f"text_ff {icfg.enhancer_inner_dim}",
        f"vis_dim {vis_dim}",
        f"vis_mlp {E}",
        f"n_views {model.num_views}",
        "ln_eps 1e-05",
    ]
    write_meta(out_dir, "fusion", meta)
    n = arena.write(os.path.join(out_dir, "fusion.bin"))
    log(f"  fusion.bin {n * 4 / 1e6:.1f} MB")


def dump_head(model, out_dir):
    """State projection + the pre-norm ACT decoder + the action MLP."""
    acfg = model.config.action
    D = model.config.interaction.hidden_dim
    head = model.action_head
    arena = Arena()

    sp = head.state_projection
    layernorm(arena, sp.net[0])                        # LayerNorm over state_dim
    linear(arena, sp.net[1], (acfg.state_hidden_dim, acfg.state_dim))
    linear(arena, sp.net[4], (acfg.num_state_tokens * D, acfg.state_hidden_dim))
    arena.add(sp.position, (1, acfg.num_state_tokens, D))
    layernorm(arena, sp.output_norm)

    arena.add(head.decoder.action_queries.weight, (acfg.horizon, D))
    for layer in head.decoder.decoder.layers:
        mha(arena, layer.self_attn, D)
        layernorm(arena, layer.norm1)
        mha(arena, layer.multihead_attn, D)
        layernorm(arena, layer.norm2)
        linear(arena, layer.linear1, (model.config.interaction.dim_feedforward, D))
        linear(arena, layer.linear2, (D, model.config.interaction.dim_feedforward))
        layernorm(arena, layer.norm3)
    # nn.TransformerDecoder(norm=None): no final LayerNorm, so nothing to export.
    for layer in head.decoder.action_projection.layers:
        linear(arena, layer)

    meta = [
        f"hidden {D}",
        f"n_layers {acfg.num_layers}",
        f"n_heads {model.config.interaction.nheads}",
        f"ff {model.config.interaction.dim_feedforward}",
        f"chunk {acfg.horizon}",
        f"action_dim {acfg.action_dim}",
        f"state_dim {acfg.state_dim}",
        f"state_tokens {acfg.num_state_tokens}",
        f"state_hidden {acfg.state_hidden_dim}",
        f"mlp_hidden {acfg.mlp_hidden_dim}",
        f"mlp_layers {head.decoder.action_projection.num_layers}",
        "ln_eps 1e-05",
    ]
    write_meta(out_dir, "head", meta)
    n = arena.write(os.path.join(out_dir, "head.bin"))
    log(f"  head.bin {n * 4 / 1e6:.1f} MB")


def write_meta(out_dir, name, lines):
    with open(os.path.join(out_dir, f"{name}.meta"), "w") as f:
        f.write("\n".join(lines) + "\n")


def dump_config(model, out_dir, image_mean, image_std, tokenizer):
    """Scalars the engine needs that are not weights: shapes, the image
    normalization, and the WordPiece ids the sub-sentence mask keys off."""
    from turbovla.evaluation import policy as ref_policy

    vcfg = model.config.vision
    tcfg = model.config.text
    cls_id, sep_id, dot_id, q_id = model.text_encoder.special_tokens
    lines = [
        f"img {vcfg.image_size}",
        f"n_views {model.num_views}",
        f"text_pad {tcfg.padding_length}",
        f"max_text_len {tcfg.max_length}",
        f"sub_sentence {int(tcfg.sub_sentence_present)}",
        f"chunk {model.chunk_size}",
        f"action_dim {model.action_dim}",
        f"state_dim {model.state_dim}",
        "img_mean " + " ".join(f"{v:.9g}" for v in image_mean),
        "img_std " + " ".join(f"{v:.9g}" for v in image_std),
        f"cls_id {cls_id}",
        f"sep_id {sep_id}",
        f"dot_id {dot_id}",
        f"question_id {q_id}",
        f"pad_id {tokenizer.pad_token_id}",
        f"unk_id {tokenizer.unk_token_id}",
        f"max_wordpiece {max(len(t) for t in tokenizer.get_vocab())}",
        f"gripper_deadband {0.0:g}",
    ]
    write_meta(out_dir, "config", lines)

    # stats.bin: proprio mean, proprio std, action min, action max -- the LIBERO
    # eval protocol from turbovla/evaluation/policy.py, not the checkpoint.
    stats = Arena()
    stats.add(ref_policy.PROPRIO_MEAN, (model.state_dim,))
    stats.add(ref_policy.PROPRIO_STD, (model.state_dim,))
    stats.add(ref_policy.ACTION_MIN, (model.action_dim,))
    stats.add(ref_policy.ACTION_MAX, (model.action_dim,))
    stats.write(os.path.join(out_dir, "stats.bin"))

    vocab = tokenizer.get_vocab()
    ordered = [None] * (max(vocab.values()) + 1)
    for token, index in vocab.items():
        ordered[index] = token
    assert all(t is not None for t in ordered), "BERT vocabulary has holes"
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ordered) + "\n")

    layout = model.config.text.padding_length_by_instruction
    with open(os.path.join(out_dir, "text_pad.txt"), "w", encoding="utf-8") as f:
        for instruction, length in sorted(layout.items()):
            f.write(f"{length}\t{instruction}\n")
    log(f"  config.txt, stats.bin, vocab.txt ({len(ordered)} tokens), "
        f"text_pad.txt ({len(layout)} instructions)")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="TurboVLA LIBERO .pth")
    p.add_argument("--out", required=True, help="output directory for the engine files")
    p.add_argument("--bert", default=os.path.join(ROOT, "build", "turbovla_ckpt", "bert"),
                   help="local bert-base-uncased directory (config + tokenizer)")
    p.add_argument("--dinov3-config",
                   default=os.path.join(HERE, "assets", "dinov3_vitb16_config.json"))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, _, _ = build_reference(args.ckpt, args.bert, args.dinov3_config)

    with open(os.path.join(os.path.dirname(args.dinov3_config),
                           "dinov3_vitb16_preprocessor.json")) as f:
        preproc = json.load(f)
    image_mean, image_std = preproc["image_mean"], preproc["image_std"]
    assert abs(preproc["rescale_factor"] - 1.0 / 255.0) < 1e-12, "unexpected rescale_factor"

    log("weights:")
    dump_vision(model, args.out)
    dump_text(model, args.out)
    dump_fusion(model, args.out)
    dump_head(model, args.out)
    dump_config(model, args.out, image_mean, image_std, model.text_encoder.tokenizer)



if __name__ == "__main__":
    sys.exit(main())
