import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

BUILD = Path(os.environ.get("VLA_TEST_BUILD", "build"))


@unittest.skipUnless(importlib.util.find_spec("diffusers") and importlib.util.find_spec("torch") and
                     (BUILD / "test_math").is_file(),
                     "requires diffusers, torch and a CMake build")
class Schedulers(unittest.TestCase):
    def test_reference_trajectories(self):
        import torch
        from diffusers import DDIMScheduler, DDPMScheduler
        for kind, cls in (("DDPM", DDPMScheduler), ("DDIM", DDIMScheduler)):
            for schedule in ("linear", "scaled_linear", "squaredcos_cap_v2"):
                for train, steps in ((1, 1), (100, 7), (100, 10), (100, 100)):
                    for clip in (0, 1):
                        with self.subTest(kind=kind, schedule=schedule, train=train, steps=steps, clip=clip):
                            output = subprocess.check_output(
                                [str(BUILD / "test_math"), schedule, str(train), str(steps), kind, str(clip)],
                                text=True, timeout=10)
                            got = np.loadtxt(io.StringIO(output), ndmin=2)
                            scheduler = cls(num_train_timesteps=train, beta_schedule=schedule, clip_sample=bool(clip))
                            scheduler.set_timesteps(steps)
                            sample = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float32)
                            noise = torch.tensor([[-0.3, 0.5, 0.1]], dtype=torch.float32)
                            expected = []
                            for i, t in enumerate(scheduler.timesteps):
                                epsilon = torch.tensor([[0.1, -0.2, 0.3]]) * np.float32((i + 1) / steps)
                                with patch("diffusers.schedulers.scheduling_ddpm.randn_tensor", return_value=noise):
                                    sample = scheduler.step(epsilon, t, sample).prev_sample
                                expected.append(sample.numpy().ravel().copy())
                            np.testing.assert_array_equal(got[:, 0], scheduler.timesteps.numpy())
                            np.testing.assert_allclose(got[:, 1:], expected, rtol=3e-4, atol=3e-4)


@unittest.skipUnless(importlib.util.find_spec("lerobot") and importlib.util.find_spec("diffusers") and
                     (BUILD / "test_math").is_file(), "requires the lerobot converter environment and a CMake build")
class DiffusionPolicy(unittest.TestCase):
    def test_exported_policy(self):
        import torch
        from diffusers import DDIMScheduler, DDPMScheduler
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import convert_diffusion as converter
        from vla_simd import policy_server

        torch.manual_seed(0)
        args = SimpleNamespace(img_h=64, img_w=64, n_cams=2, down_dims="32,64,128",
                               n_obs_steps=2, horizon=16, n_action_steps=8, crop_h=0,
                               state_dim=6, action_dim=6)
        policy, config, cameras, shape, stats = converter.build_random(args)
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (2, 2, 64, 64, 3), dtype=np.uint8)
        state = rng.uniform(-50, 50, (2, 6)).astype(np.float32)
        pixels = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float() / 255
        for i, camera in enumerate(cameras):
            pixels[:, i] = ((pixels[:, i] - torch.from_numpy(stats[camera]["mean"])[:, None, None]) /
                            torch.from_numpy(stats[camera]["std"])[:, None, None])
        batch = {"observation.state": torch.from_numpy(state / 100).unsqueeze(0),
                 "observation.images": pixels.unsqueeze(0)}
        noise = rng.standard_normal((8, 16, 6), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            weights = policy.state_dict()
            count = len(cameras) if config.use_separate_rgb_encoder_per_camera else 1
            for i in range(count):
                prefix = f"diffusion.rgb_encoder.{i}" if count > 1 else "diffusion.rgb_encoder"
                converter.dump_rgb_encoder(weights, prefix, directory, f"rgb_encoder{i}",
                                           16 if config.use_group_norm else 0)
            converter.dump_unet(weights, directory, config)
            converter.dump_stats(stats, cameras, directory)
            for kind, cls in (("DDPM", DDPMScheduler), ("DDIM", DDIMScheduler)):
                with self.subTest(scheduler=kind):
                    converter.dump_meta(config, cameras, shape, directory, kind, 7)
                    policy.diffusion.noise_scheduler = cls(
                        num_train_timesteps=config.num_train_timesteps, beta_start=config.beta_start,
                        beta_end=config.beta_end, beta_schedule=config.beta_schedule,
                        clip_sample=config.clip_sample, clip_sample_range=config.clip_sample_range,
                        prediction_type=config.prediction_type)
                    policy.diffusion.num_inference_steps = 7
                    draws = [torch.from_numpy(n).unsqueeze(0) for n in noise[1:]]
                    with torch.inference_mode(), patch(
                            "diffusers.schedulers.scheduling_ddpm.randn_tensor", side_effect=draws):
                        expected = policy.diffusion.generate_actions(
                            batch, noise=torch.from_numpy(noise[0].copy()).unsqueeze(0))[0].numpy()
                    engine = policy_server.DiffusionEngine(SimpleNamespace(
                        lib=str(BUILD / f"libvla_simd_diffusion{policy_server.LIB_EXT}"), model_dir=directory))
                    try:
                        actual = engine._run(frames.ctypes.data_as(policy_server.U8P),
                                             state.ctypes.data_as(policy_server.F32P), 0,
                                             after=(noise.ctypes.data_as(policy_server.F32P),))
                        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
                    finally:
                        engine.close()


if __name__ == "__main__":
    unittest.main()
