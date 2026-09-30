import concurrent.futures
import io
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import _gguf
import convert_hf_safetensors
from vla_simd import gguf_stage, policy_server


class Benchmark(unittest.TestCase):
    def test_json_and_invalid_timing(self):
        engine = SimpleNamespace(
            chunk=1, lib=SimpleNamespace(vla_backend_name=Mock(return_value=b"scalar"),
                                        vla_int8_available=Mock(return_value=0)),
            describe=Mock(return_value="test configuration"), predict=Mock(), close=Mock(),
            warmup_input=Mock(return_value=(np.zeros((1, 1, 1, 3), np.uint8),)))
        argv = ["serve", "--model", "act", "--model-dir", "test.gguf", "--bench", "2", "--json"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(sys, "stdout", output), \
             patch.object(gguf_stage, "stage", return_value="staged"), \
             patch.object(policy_server.MODELS["act"], "engine_cls", return_value=engine):
            policy_server.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["checkpoint"], "test.gguf")
        self.assertEqual(result["queries"], 2)
        self.assertGreater(result["peak_rss_bytes"], 0)
        self.assertGreater(result["baseline_rss_bytes"], 0)
        self.assertLessEqual(result["baseline_rss_bytes"], result["peak_rss_bytes"])
        self.assertEqual(result["warmup"], 2)
        self.assertGreaterEqual(result["p95_ms"], result["median_ms"])
        self.assertEqual(engine.predict.call_count, 4)
        engine.close.assert_called_once()
        engine.predict.reset_mock()
        output = io.StringIO()
        with patch.object(sys, "argv", argv + ["--warmup", "5"]), patch.object(sys, "stdout", output), \
             patch.object(gguf_stage, "stage", return_value="staged"), \
             patch.object(policy_server.MODELS["act"], "engine_cls", return_value=engine):
            policy_server.main()
        self.assertEqual(json.loads(output.getvalue())["warmup"], 5)
        self.assertEqual(engine.predict.call_count, 7)
        for option, values in (("--soak", ("nan", "inf")), ("--obs-queue-timeout", ("nan", "inf")),
                               ("--warmup", ("0",))):
            for value in values:
                with patch.object(sys, "argv", argv + [option, value]), \
                     patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                    policy_server.main()
                self.assertEqual(error.exception.code, 2)


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.path = self.root / "model.gguf"

    def pack(self, text="chunk 2\n", layout=None):
        (self.source / "config.txt").write_text(text)
        (self.source / "weights.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 4))
        _gguf.pack(self.source, self.path, "act", "test", layout or {})

    def test_roundtrip_and_parts(self):
        self.pack(layout={"weights.bin": [("F32", 8), ("BF16", 8)]})
        files = _gguf.read_files(self.path)
        self.assertEqual(files, {p.name: p.read_bytes() for p in self.source.iterdir()})
        whole = self.path.read_bytes()
        for cut in (3, 20, len(whole) // 2, len(whole) - 1):
            self.path.write_bytes(whole[:cut])
            with self.assertRaises(ValueError):
                _gguf.read_files(self.path)

    def test_reject_bad_layout_preserves_destination(self):
        self.pack()
        whole = self.path.read_bytes()
        for layout in ([('F32', -4), ('F32', 20)], [('F32', 20)], [('F32', 4)]):
            with self.assertRaises(ValueError):
                _gguf.pack(self.source, self.path, "act", "test", {"weights.bin": layout})
            self.assertEqual(self.path.read_bytes(), whole)

    def test_pack_publication(self):
        self.pack()
        whole = self.path.read_bytes()
        with patch.object(_gguf.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.pack("changed\n")
        self.assertEqual(self.path.read_bytes(), whole)
        self.assertEqual({p.name for p in self.root.iterdir()}, {"source", "model.gguf"})
        replace = os.replace
        barrier = threading.Barrier(4)

        def publish(source, destination):
            barrier.wait(timeout=5)
            replace(source, destination)

        with patch.object(_gguf.os, "replace", publish), concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            writes = [pool.submit(_gguf.pack, self.source, self.path, "act", "test", {}) for _ in range(4)]
            for future in writes:
                future.result(timeout=10)
        self.assertEqual(_gguf.read_files(self.path)["config.txt"], b"changed\n")
        self.assertEqual({p.name for p in self.root.iterdir()}, {"source", "model.gguf"})

    def test_metadata_bounds(self):
        for value in (struct.pack("<Q", 100), struct.pack("<Q", 1) + b"x" + struct.pack("<IIQ", 9, 0, 999)):
            self.path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + value)
            with self.assertRaises(ValueError):
                gguf_stage.read_metadata(self.path)

    @unittest.skipUnless((Path(os.environ.get("VLA_TEST_BUILD", "build")) / "vla-simd-gguf").is_file(),
                         "requires a CMake build")
    def test_extraction_refuses_symlinks_and_overwrite(self):
        self.pack()
        nested = self.source / "nested" / "deeper"
        nested.mkdir(parents=True)
        (nested / "config.txt").write_text("nested")
        _gguf.pack(self.source, self.path, "act", "test", {})
        binary = Path(os.environ.get("VLA_TEST_BUILD", "build")) / "vla-simd-gguf"

        def extract(out):
            return subprocess.run([str(binary), "extract", str(self.path), str(out)],
                                  capture_output=True, timeout=10)

        target = self.root / "target"
        self.assertEqual(extract(target).returncode, 0)
        self.assertEqual((target / "nested/deeper/config.txt").read_text(), "nested")
        self.assertNotEqual(extract(target).returncode, 0)
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        self.assertNotEqual(extract(link).returncode, 0)
        for name in ("config.txt", "nested"):
            output = self.root / ("out-" + name)
            output.mkdir()
            (output / name).symlink_to(target / name, target_is_directory=name == "nested")
            self.assertNotEqual(extract(output).returncode, 0)
        self.assertEqual((target / "config.txt").read_text(), "chunk 2\n")

    def test_cache_replacement_and_concurrency(self):
        self.pack()
        cache = self.root / "cache"
        first = gguf_stage.stage(str(self.path), "act", str(cache))
        self.pack("chunk 3\n")
        second = gguf_stage.stage(str(self.path), "act", str(cache))
        self.assertNotEqual(first, second)
        self.assertEqual((Path(second) / "config.txt").read_text(), "chunk 3\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda _: gguf_stage.stage(str(self.path), "act", str(self.root / "parallel")), range(16)))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual((Path(paths[0]) / "config.txt").read_text(), "chunk 3\n")
        (self.root / "config.txt").write_text("chunk 4\n")
        third = gguf_stage.stage(str(self.path), "act", str(cache))
        self.assertEqual((Path(third) / "config.txt").read_text(), "chunk 4\n")

    def test_interrupted_staging_retries(self):
        self.pack()
        cache = self.root / "cache"
        populate = gguf_stage._populate

        def fail(gguf, out, *args):
            (Path(out) / "config.txt").write_text("partial")
            raise OSError("interrupted")

        with patch.object(gguf_stage, "_populate", fail), self.assertRaises(OSError):
            gguf_stage.stage(str(self.path), "act", str(cache))
        self.assertEqual(list(cache.iterdir()), [])
        with patch.object(gguf_stage, "_populate", populate):
            path = gguf_stage.stage(str(self.path), "act", str(cache))
        self.assertEqual((Path(path) / "config.txt").read_text(), "chunk 2\n")


class Inputs(unittest.TestCase):
    def test_bf16_nan_and_rounding(self):
        bits = np.array([0x7F800001, 0x7FFFFFFF, 0xFFFFFFFF, 0x3F808000, 0x3F818000], np.uint32)
        got = convert_hf_safetensors.f32_to_bf16(bits.view(np.float32))
        self.assertTrue(np.all((got[:3] & 0x7F80) == 0x7F80))
        self.assertTrue(np.all((got[:3] & 0x7F) != 0))
        np.testing.assert_array_equal(got[3:], [0x3F80, 0x3F82])

    def test_safetensors_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [-4, 4]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
            reader = convert_hf_safetensors.Safetensors(path)
            try:
                with self.assertRaises(ValueError):
                    reader.bits("w")
            finally:
                reader.mm.close()
                reader.f.close()
            path.write_bytes(struct.pack("<Q", 1 << 63))
            with self.assertRaises(ValueError):
                convert_hf_safetensors.Safetensors(path)

    def test_history_padding(self):
        history = policy_server.History(2)
        values, mask = history.push(1)
        self.assertEqual(values, [1, 1])
        np.testing.assert_array_equal(mask, [0, 1])
        self.assertEqual(history.push(2)[0], [1, 2])
        self.assertEqual(history.push(3)[0], [2, 3])
        with self.assertRaises(ValueError):
            policy_server.History(0)

    def test_nonfinite_observations(self):
        engine = object.__new__(policy_server._Engine)
        engine.state_dim = 2
        for state in ([1, np.nan], [np.inf, 1], [1]):
            with self.assertRaises(ValueError):
                engine._state(state)
        for frame in (np.empty((0, 2, 3), np.uint8), np.full((2, 2, 3), np.nan)):
            with self.assertRaises(ValueError):
                policy_server.ObservationAdapter._as_uint8(frame)

    def test_pickle_rejects_globals(self):
        with self.assertRaises(pickle.UnpicklingError):
            policy_server._loads(b"cos\nsystem\n(S'false'\ntR.")

    def test_engine_rejects_invalid_results_and_closed_handle(self):
        engine = object.__new__(policy_server._Engine)
        engine.chunk = engine.action_dim = 1
        engine.name = "test"
        engine.h = 1
        engine.lock = threading.Lock()

        def predict(handle, output):
            output[0] = np.nan
            return 0

        engine._fn = lambda _: predict
        with self.assertRaisesRegex(RuntimeError, "nonfinite actions"):
            engine._run()
        engine.h = None
        with self.assertRaisesRegex(RuntimeError, "engine is closed"):
            engine._run()


if __name__ == "__main__":
    unittest.main()
