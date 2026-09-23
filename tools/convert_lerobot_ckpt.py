#!/usr/bin/env python3
"""
convert_lerobot_ckpt.py

Convert a finetuned lerobot SmolVLA checkpoint into the flat .meta/.bin weight
format of the vla.simd engine, so vla_simd/policy_server.py can run it.

Same output layout as convert_hf_safetensors.py (which converts the HuggingFaceVLA
base models), but reads the checkpoint through the lerobot factories a torch
policy server would use, so everything checkpoint-specific comes from the
deployment path rather than from a base model's defaults. For the UR10e finetune
that means:
  * 16 VLM layers / expert_h 720 / expert_ffn 2048, not smolvla_libero's 32/480/1280
  * 2 camera views (a third is declared but never fed, empty_cameras=0)
  * 7-dim state/action MEAN_STD statistics (config.json still says 6)
  * the SmolVLM2 tokenizer named by the checkpoint's tokenizer_processor

Run in the venv the checkpoint was trained with (lerobot + torch + the pinned
transformers), from the repo root:
  ~/work/smolvla-sim/.venv/bin/python tools/convert_lerobot_ckpt.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
except ImportError as e:
    sys.exit(f"Missing dep: {e}. Run inside a venv with lerobot installed.")

DEFAULT_CHECKPOINT = "~/work/ur/smolvla-ur/outputs/smolvla_ur10e_1.1.0-fix/checkpoints/020000"
ROPE_BASE = 10000.0   # apply_rope's hardcoded default, NOT text_config.rope_theta


def resolve_checkpoint(path: str) -> Path:
    """Accept a pretrained_model dir, a step dir, or the whole run dir (latest step)."""
    p = Path(path).expanduser()
    if (p / "model.safetensors").is_file():
        return p
    if (p / "pretrained_model" / "model.safetensors").is_file():
        return p / "pretrained_model"
    ckpt_root = p / "checkpoints" if (p / "checkpoints").is_dir() else p
    steps = sorted(d for d in ckpt_root.glob("*") if d.is_dir() and d.name.isdigit())
    if steps and (steps[-1] / "pretrained_model" / "model.safetensors").is_file():
        return steps[-1] / "pretrained_model"
    sys.exit(f"No SmolVLA checkpoint found under {p}")


def w(t):
    """tensor -> contiguous fp32 numpy (bf16 checkpoint values widen exactly)."""
    return np.ascontiguousarray(t.detach().cpu().float().numpy().astype(np.float32))


def wb(t):
    """tensor -> bf16 as uint16. Lossless for the bf16 towers, rounds fp32 tensors."""
    return t.detach().cpu().to(torch.bfloat16).view(torch.uint16).contiguous().numpy()


def stats_of(pipeline, key, ftype):
    """(mean, std) as recorded in the checkpoint's normalization statistics."""
    for step in pipeline.steps:
        s = getattr(step, "stats", None)
        if s and key in s and s[key].get("mean") is not None:
            mode = step.norm_map.get(ftype, "IDENTITY")
            if mode != "MEAN_STD":
                sys.exit(f"{key} is normalized {getattr(mode, 'value', mode)}; "
                         "the engine implements MEAN_STD only")
            return (np.asarray(torch.as_tensor(s[key]["mean"]).reshape(-1).float().cpu(), np.float32),
                    np.asarray(torch.as_tensor(s[key]["std"]).reshape(-1).float().cpu(), np.float32))
    sys.exit(f"No normalization statistics for {key!r} in the checkpoint.")


def pipeline_tokenizer(pipeline):
    for step in pipeline.steps:
        tk = getattr(step, "input_tokenizer", None)
        if tk is not None:
            return tk
    sys.exit("No tokenizer_processor step in the checkpoint's preprocessor.")


# ---------------------------------------------------------------------------
# weight export - byte layouts must match the engine's loaders
# (src/models/smolvla/{smollm2_lm,siglip_vision,action_expert,smolvla_model}.cpp)
# ---------------------------------------------------------------------------

def dump_vlm(vlmx, out, n_layers):
    tc = vlmx.config.text_config
    tm = vlmx.get_vlm_model().text_model
    with open(f"{out}/vlm.meta", "w") as f:
        f.write(f"hidden {tc.hidden_size}\nn_q {tc.num_attention_heads}\n")
        f.write(f"n_kv {tc.num_key_value_heads}\nhead_dim {tc.head_dim}\nffn {tc.intermediate_size}\n")
        f.write(f"eps {tc.rms_norm_eps}\nrope_base {ROPE_BASE}\nn_layers {n_layers}\n")
    with open(f"{out}/vlm.bin", "wb") as f:
        for L in range(n_layers):                   # fp32 region: per-layer norms
            ly = tm.layers[L]
            w(ly.input_layernorm.weight).tofile(f)
            w(ly.post_attention_layernorm.weight).tofile(f)
        w(tm.norm.weight).tofile(f)                 # output norm
        for L in range(n_layers):                   # bf16 region: matmul weights
            ly = tm.layers[L]
            wb(ly.self_attn.q_proj.weight).tofile(f)
            wb(ly.self_attn.k_proj.weight).tofile(f)
            wb(ly.self_attn.v_proj.weight).tofile(f)
            wb(ly.self_attn.o_proj.weight).tofile(f)
            wb(ly.mlp.gate_proj.weight).tofile(f)
            wb(ly.mlp.up_proj.weight).tofile(f)
            wb(ly.mlp.down_proj.weight).tofile(f)


@torch.no_grad()
def capture_position_ids(vlmx, img_size):
    """The SigLIP position-id per patch, as the installed transformers computes it.

    SmolVLM buckets fractional patch coordinates instead of indexing 0..n-1, and the
    formula was shifted in transformers 4.55-4.57 and restored in 5.0 (`k/n*(1-1e-6)`
    replaced `arange(0, 1-1e-6, 1/n)`), which lands every coordinate one bucket lower:
    32 patches per side map to rows [0,0,1,...,30] instead of [0,...,31].
    This checkpoint was trained and is served on 4.57.6, so that is the mapping to
    reproduce. The engine adds position row p to patch p, so the mapping is baked
    into the dumped table rather than assumed on either side.
    """
    vm = vlmx.get_vlm_model().vision_model
    seen = {}

    def grab(_module, inputs):     # returning a value here would replace the input
        seen["ids"] = np.asarray(inputs[0].detach().cpu()).reshape(-1)

    hook = vm.embeddings.position_embedding.register_forward_pre_hook(grab)
    vm(pixel_values=torch.zeros(1, 3, img_size, img_size), patch_attention_mask=None)
    hook.remove()
    return seen["ids"]


def dump_vit(vlmx, out, n_img_tok, hidden):
    vm = vlmx.get_vlm_model().vision_model
    conn = vlmx.get_vlm_model().connector
    vc = vlmx.config.vision_config
    nl, patch, img = vc.num_hidden_layers, vc.patch_size, vc.image_size
    n_patches = (img // patch) ** 2
    sf = vlmx.get_vlm_model().config.scale_factor
    pos_ids = capture_position_ids(vlmx, img)
    if pos_ids.shape[0] != n_patches:
        sys.exit(f"position ids {pos_ids.shape[0]} != {n_patches} patches")
    pos_mode = 'shifted' if pos_ids[1] != 1 else 'identity'
    print(f"  siglip position ids: {pos_ids[:4].tolist()} ... {pos_ids[-2:].tolist()} "
          f"({pos_mode})")
    with open(f"{out}/vit.meta", "w") as f:
        f.write(f"hidden {vc.hidden_size}\nn_heads {vc.num_attention_heads}\n")
        f.write(f"head_dim {vc.hidden_size // vc.num_attention_heads}\ninter {vc.intermediate_size}\n")
        f.write(f"n_layers {nl}\npatch {patch}\nimg {img}\nn_patches {n_patches}\n")
        f.write(f"ln_eps {vc.layer_norm_eps}\nscale_factor {sf}\nmm_out {hidden}\nn_img_tok {n_img_tok}\n")
    with open(f"{out}/vit.bin", "wb") as f:
        w(vm.embeddings.patch_embedding.bias).tofile(f)
        w(vm.embeddings.position_embedding.weight)[pos_ids].tofile(f)   # gathered to patch order
        for L in range(nl):
            ly = vm.encoder.layers[L]
            w(ly.layer_norm1.weight).tofile(f);     w(ly.layer_norm1.bias).tofile(f)
            w(ly.self_attn.q_proj.bias).tofile(f);  w(ly.self_attn.k_proj.bias).tofile(f)
            w(ly.self_attn.v_proj.bias).tofile(f);  w(ly.self_attn.out_proj.bias).tofile(f)
            w(ly.layer_norm2.weight).tofile(f);     w(ly.layer_norm2.bias).tofile(f)
            w(ly.mlp.fc1.bias).tofile(f);           w(ly.mlp.fc2.bias).tofile(f)
        w(vm.post_layernorm.weight).tofile(f);      w(vm.post_layernorm.bias).tofile(f)
        wb(vm.embeddings.patch_embedding.weight).tofile(f)   # [768,3,16,16] C-order = [768,768]
        for L in range(nl):
            ly = vm.encoder.layers[L]
            wb(ly.self_attn.q_proj.weight).tofile(f)
            wb(ly.self_attn.k_proj.weight).tofile(f)
            wb(ly.self_attn.v_proj.weight).tofile(f)
            wb(ly.self_attn.out_proj.weight).tofile(f)
            wb(ly.mlp.fc1.weight).tofile(f)
            wb(ly.mlp.fc2.weight).tofile(f)
        wb(conn.modality_projection.proj.weight).tofile(f)

    return pos_mode

def dump_aex(model, vlmx, cfg, out, n_layers, san):
    aex = vlmx.lm_expert
    tc = vlmx.config.text_config
    expert_h = vlmx.expert_hidden_size
    expert_ffn = aex.layers[0].mlp.gate_proj.weight.shape[0]
    with open(f"{out}/aex.meta", "w") as f:
        f.write(f"expert_h {expert_h}\nexpert_ffn {expert_ffn}\nn_q {tc.num_attention_heads}\n")
        f.write(f"n_kv {tc.num_key_value_heads}\nhead_dim {tc.head_dim}\n")
        f.write(f"eps {tc.rms_norm_eps}\nrope_base {ROPE_BASE}\nn_layers {n_layers}\n")
        f.write(f"self_attn_every_n {san}\n")
        f.write(f"chunk {cfg.chunk_size}\nnum_steps {cfg.num_steps}\n")
        f.write(f"max_action_dim {cfg.max_action_dim}\n")
        f.write(f"min_period {cfg.min_period}\nmax_period {cfg.max_period}\n")
    with open(f"{out}/aex.bin", "wb") as f:
        for L in range(n_layers):
            ly = aex.layers[L]
            w(ly.input_layernorm.weight).tofile(f)
            w(ly.self_attn.q_proj.weight).tofile(f)
            w(ly.self_attn.k_proj.weight).tofile(f)
            w(ly.self_attn.v_proj.weight).tofile(f)
            w(ly.self_attn.o_proj.weight).tofile(f)
            w(ly.post_attention_layernorm.weight).tofile(f)
            w(ly.mlp.gate_proj.weight).tofile(f)
            w(ly.mlp.up_proj.weight).tofile(f)
            w(ly.mlp.down_proj.weight).tofile(f)
        w(aex.norm.weight).tofile(f)
        w(model.action_in_proj.weight).tofile(f);      w(model.action_in_proj.bias).tofile(f)
        w(model.action_time_mlp_in.weight).tofile(f);  w(model.action_time_mlp_in.bias).tofile(f)
        w(model.action_time_mlp_out.weight).tofile(f); w(model.action_time_mlp_out.bias).tofile(f)
        w(model.action_out_proj.weight).tofile(f);     w(model.action_out_proj.bias).tofile(f)


def dump_tokenizer(tokenizer, out):
    """Flat vocab/merges for the engine's byte-level BPE tokenizer."""
    tj = json.loads(tokenizer.backend_tokenizer.to_str())
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/vocab.txt", "w", encoding="utf-8") as f:
        for tok, i in vocab.items():
            f.write(f"{i}\t{tok}\n")
    with open(f"{out}/merges.txt", "w", encoding="utf-8") as f:
        for m in merges:
            f.write((m if isinstance(m, str) else f"{m[0]} {m[1]}") + "\n")
    return len(vocab), len(merges)


def camera_keys(pre, cfg):
    """The image keys the deployment path is actually fed, in order.

    A finetune declares one PolicyFeature per camera slot (camera1..camera3) but
    the preprocessor's rename_observations_processor maps the real camera names
    onto only the slots that are ever populated - the rest stay empty. Without
    a rename step the declared image features are already the real keys.
    """
    rename = rename_map(pre)
    return [k for k in cfg.image_features if not rename or k in rename.values()]


def rename_map(pre):
    """The preprocessor's {real camera key -> declared slot} map, or {} if absent."""
    for step in getattr(pre, "steps", []):
        rename = getattr(getattr(step, "config", None), "rename_map", None)
        if rename is None:
            rename = getattr(step, "rename_map", None)
        if rename:
            return dict(rename)
    return {}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT,
                   help=f"Run dir, step dir, or pretrained_model dir (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--out", default=None,
                   help="Output dir for the .meta/.bin weights (default: <checkpoint>/simd)")
    p.add_argument("--task", default="pick up the cup",
                   help="Instruction written to config.txt")
    args = p.parse_args()

    ckpt = resolve_checkpoint(args.ckpt)
    out = os.path.expanduser(args.out) if args.out else str(ckpt.parent / "simd")
    os.makedirs(out, exist_ok=True)

    print(f"Loading {ckpt} on cpu ...")
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    for k in ("add_image_special_tokens", "adapt_to_pi_aloha"):
        if getattr(cfg, k, False):
            sys.exit(f"{k}=True is not implemented by the engine")
    san = cfg.self_attn_every_n_layers if "cross" in cfg.attention_mode else 1
    if san <= 0:
        sys.exit(f"attention_mode={cfg.attention_mode} with self_attn_every_n_layers={san} "
                 "(all-cross expert) is not supported")
    cfg.device = "cpu"
    cfg.use_amp = False
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    # fp32 is the engine's numerical target; the checkpoint's towers are bf16.
    policy.model.float()

    model = policy.model
    vlmx = model.vlm_with_expert
    n_layers = vlmx.num_vlm_layers
    hidden = vlmx.config.text_config.hidden_size
    print(f"  vlm layers={n_layers} hidden={hidden} expert_h={vlmx.expert_hidden_size} "
          f"attn={vlmx.attention_mode} self_attn_every_n={vlmx.self_attn_every_n_layers}")

    with torch.no_grad():
        probe = torch.zeros(1, 3, cfg.resize_imgs_with_padding[0], cfg.resize_imgs_with_padding[1])
        n_img_tok = vlmx.embed_image(probe).shape[1]
    print(f"  image tokens per view: {n_img_tok}")

    print("Dumping weights ...")
    dump_vlm(vlmx, out, n_layers)
    pos_mode = dump_vit(vlmx, out, n_img_tok, hidden)
    dump_aex(model, vlmx, cfg, out, n_layers, san)

    tm = vlmx.get_vlm_model().text_model
    wb(tm.embed_tokens.weight).tofile(f"{out}/emb.bin")
    with open(f"{out}/heads.bin", "wb") as f:
        w(model.state_proj.weight).tofile(f)
        w(model.state_proj.bias).tofile(f)

    smean, sstd = stats_of(pre, "observation.state", "STATE")
    amean, astd = stats_of(post, "action", "ACTION")
    smean.tofile(f"{out}/stats_state_mean.bin"); sstd.tofile(f"{out}/stats_state_std.bin")
    amean.tofile(f"{out}/stats_action_mean.bin"); astd.tofile(f"{out}/stats_action_std.bin")
    print(f"  state stats {smean.shape[0]}-dim, action stats {amean.shape[0]}-dim")

    # The preprocessor's rename map, not the declared image features: a finetune
    # declares more camera slots than the deployment path is ever fed.
    n_views = len(camera_keys(pre, cfg))

    with open(f"{out}/heads.meta", "w") as f:
        f.write(f"vocab {tm.embed_tokens.weight.shape[0]}\nhidden {hidden}\n")
        f.write(f"max_state_dim {cfg.max_state_dim}\n")
        f.write(f"real_state_dim {smean.shape[0]}\nreal_action_dim {amean.shape[0]}\n")
        f.write(f"n_views {n_views}\n")

    tokenizer = pipeline_tokenizer(pre)
    nv, nm = dump_tokenizer(tokenizer, f"{out}/tok")
    print(f"  tokenizer {tokenizer.name_or_path}: vocab={nv} merges={nm}")

    with open(f"{out}/config.txt", "w") as f:
        f.write(f"instruction {args.task}\n")
        f.write(f"tokenizer_max_length {cfg.tokenizer_max_length}\n")
        f.write(f"pad_token_id {tokenizer.pad_token_id}\n")
        f.write(f"chunk {cfg.chunk_size}\nnum_steps {cfg.num_steps}\nn_views {n_views}\n")
        f.write(f"checkpoint {ckpt}\n")
        # Which SigLIP position-id convention got baked into vit.bin. The embedding is
        # dumped already gathered to patch order, so a consumer that assumes the other
        # convention reproduces the whole vision tower wrongly with no local symptom.
        f.write(f"pos_ids {pos_mode}\n")
        rename = rename_map(pre)
        for i, dst in enumerate(camera_keys(pre, cfg)):
            f.write(f"cam{i} {dst.split('.')[-1] if '.' in dst else dst}\n")
        for src, dst in rename.items():
            f.write(f"rename {src.split('.')[-1]} {dst.split('.')[-1]}\n")

    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)
                if os.path.isfile(os.path.join(out, f)))
    print(f"done -> {out} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
