import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from vla_simd import gguf_stage, policy_server


def check_turbovla(args):
    import convert_turbovla as converter
    from huggingface_hub import snapshot_download

    if args.reference_code:
        converter.REF = args.reference_code
    bert = snapshot_download("google-bert/bert-base-uncased", allow_patterns=["*.json", "vocab.txt"])
    reference, _, _ = converter.build_reference(
        args.checkpoint, bert, str(Path(__file__).parent / "assets/dinov3_vitb16_config.json"))
    stats_path = args.stats or str(Path(converter.REF) / "experiments/libero/configs/libero_all4_stats.json")
    stats = json.loads(Path(stats_path).read_text())["libero_all4_no_noops"]
    engine = policy_server.TurboVlaEngine(SimpleNamespace(
        lib=str(Path(args.build) / f"libvla_simd_turbovla{policy_server.LIB_EXT}"),
        model_dir=gguf_stage.stage(args.gguf, "turbovla", args.cache)))
    rng = np.random.default_rng(0)
    mean = np.array(stats["proprio"]["mean"], np.float32)
    std = np.array(stats["proprio"]["std"], np.float32)
    action_min = np.array(stats["action"]["min"], np.float32)
    action_max = np.array(stats["action"]["max"], np.float32)
    errors = []
    try:
        for i in range(args.samples):
            task = args.task if i % 2 == 0 else "put the red cup on the table"
            frames = rng.integers(0, 256, (engine.n_views, engine.img_size, engine.img_size, 3), dtype=np.uint8)
            state = mean + rng.standard_normal(engine.state_dim).astype(np.float32) * std
            pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).float() * (1 / 255)
            pixels = ((pixels - torch.tensor([.485, .456, .406])[None, :, None, None]) /
                      torch.tensor([.229, .224, .225])[None, :, None, None])
            with torch.inference_mode():
                expected = reference([task], pixels.unsqueeze(0),
                                     torch.from_numpy((state - mean) / (std + 1e-6)).unsqueeze(0))[0].numpy()
            actual = engine._run(frames.ctypes.data_as(policy_server.U8P),
                                 state.ctypes.data_as(policy_server.F32P), task.encode(), 0)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
            raw = engine.predict((frames, state, task), 0)
            expected_raw = expected.copy()
            expected_raw[:, :6] = .5 * (expected[:, :6] + 1) * (action_max[:6] - action_min[:6]) + action_min[:6]
            expected_raw[:, 6] = np.where(expected[:, 6] < 0, -1, 1)
            np.testing.assert_allclose(raw, expected_raw, rtol=2e-4, atol=2e-4)
            errors.append(float(np.max(np.abs(actual - expected))))
    finally:
        engine.close()
    print(json.dumps({"samples": args.samples, "max_normalized_error": max(errors)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("gguf")
    parser.add_argument("--model", choices=("act", "impact", "turbovla"), default="act")
    parser.add_argument("--task", default="Put the tape into the box")
    parser.add_argument("--build", default="build")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--reference-code")
    parser.add_argument("--stats")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.model == "turbovla":
        return check_turbovla(args)
    if args.model == "act":
        from lerobot.policies.act.configuration_act import ACTConfig as Config
        from lerobot.policies.act.modeling_act import ACTPolicy as Policy
    else:
        from lerobot.policies.impact.configuration_impact import IMPACTConfig as Config
        from lerobot.policies.impact.modeling_impact import IMPACTPolicy as Policy
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    config = Config.from_pretrained(args.checkpoint)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    reference = Policy.from_pretrained(args.checkpoint, config=config).eval()
    stats = {}
    for path in sorted(Path(args.checkpoint).glob("*normalizer*.safetensors")):
        stats.update(load_file(path))
    engine = policy_server.MODELS[args.model].engine_cls(SimpleNamespace(
        lib=str(Path(args.build) / f"libvla_simd_{args.model}{policy_server.LIB_EXT}"),
        model_dir=gguf_stage.stage(args.gguf, args.model, args.cache)))
    rng = np.random.default_rng(0)
    errors = []
    try:
        for _ in range(args.samples):
            frames = rng.integers(0, 256, (engine.n_cams, engine.img_h, engine.img_w, 3), dtype=np.uint8)
            state = np.ascontiguousarray(stats["observation.state.mean"].numpy() +
                                         rng.standard_normal(engine.state_dim).astype(np.float32))
            batch = {}
            values = {"observation.state": torch.from_numpy(state).unsqueeze(0)}
            values.update((key, torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255)
                          for key, frame in zip(config.image_features, frames))
            for key, value in values.items():
                batch[key] = (value - stats[f"{key}.mean"]) / (stats[f"{key}.std"] + 1e-8)
            if args.model == "impact":
                from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
                tokens = tokenizer([args.task], padding="max_length", truncation=True,
                                   max_length=config.tokenizer_max_length, return_tensors="pt")
                batch[OBS_LANGUAGE_TOKENS] = tokens.input_ids
                batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens.attention_mask
            with torch.inference_mode():
                expected = reference.predict_action_chunk(batch)[0].numpy()
            actual = engine._run(frames.ctypes.data_as(policy_server.U8P),
                                 state.ctypes.data_as(policy_server.F32P),
                                 *((args.task.encode(),) if args.model == "impact" else ()), 0)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
            raw = engine.predict((frames, state, args.task) if args.model == "impact" else (frames, state), 0)
            expected_raw = expected * stats["action.std"].numpy() + stats["action.mean"].numpy()
            np.testing.assert_allclose(raw, expected_raw, rtol=2e-4, atol=1e-2)
            errors.append(float(np.max(np.abs(actual - expected))))
    finally:
        engine.close()
    print(json.dumps({"samples": args.samples, "max_normalized_error": max(errors)}))


if __name__ == "__main__":
    main()
