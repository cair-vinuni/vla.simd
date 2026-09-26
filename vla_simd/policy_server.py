#!/usr/bin/env python3
"""
policy_server.py

One server for every policy in the tree:

    vla-simd-serve --model act       --model-dir act.gguf
    vla-simd-serve --model octo      --model-dir octo.gguf
    vla-simd-serve --model impact    --model-dir impact.gguf
    vla-simd-serve --model turbovla  --model-dir turbovla.gguf
    vla-simd-serve --model smolvla   --model-dir smolvla.gguf
    vla-simd-serve --model diffusion --model-dir diffusion.gguf

--model-dir is a .gguf, a directory holding one, or hf://<user>/<repo>[/<file>.gguf].

It speaks lerobot's async-inference protocol and is a drop-in replacement for
`lerobot.async_inference.policy_server`: same gRPC service, same messages, same
pickled payloads. Only what happens between receiving an observation and
returning a chunk differs -- no torch policy is built, nothing is moved to a
device, and inference runs in libvla_simd_<model>.so on the CPU.

What differs per policy is the observation contract, and that is all a ModelSpec
carries: ACT and Diffusion take no instruction, Octo takes no proprioceptive
state, Octo and Diffusion take a window of observations rather than one, and
TurboVLA consumes its frames at the checkpoint's resolution without resampling.
The service, queueing and CLI are shared.

The checkpoint is a GGUF converted ahead of time, so `pretrained_name_or_path`
in the client's policy instructions is not fetched: the server serves --model-dir and
warns if the client asked for something else. `actions_per_chunk` is honoured by
truncating the chunk.

lerobot supplies the wire dataclasses (TimedObservation / TimedAction /
RemotePolicyConfig) and the raw-observation -> feature mapping, so key handling
matches the reference server instead of drifting from a re-implementation. That
pulls in torch, but no model, no CUDA context and no GPU memory.

Per-model differences live in a ModelSpec; the service, queueing and CLI are
shared.

Protocol (see lerobot/transport/services.proto):
    Ready(Empty) -> Empty
    SendPolicyInstructions(PolicySetup) -> Empty     pickled RemotePolicyConfig
    SendObservations(stream Observation) -> Empty    pickled TimedObservation
    GetActions(Empty) -> Actions                     pickled list[TimedAction]

The port is unauthenticated: keep it on a lab LAN or an SSH tunnel.
"""

import argparse
import ctypes
import io
import json
import logging
import math
import os
import pickle  # nosec: the lerobot async-inference protocol is pickle-based
import sys
import threading
import time
from collections import deque
from concurrent import futures
from queue import Empty, Full, Queue

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_EXT = ".dylib" if sys.platform == "darwin" else ".so"
LIB_DIRS = (*(sys.modules[__package__].__path__ if __package__ else ()), os.path.join(HERE, "build"))

U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int32)

logger = logging.getLogger("vla_simd_policy_server")

VLA_ABI_VERSION = 1


class _WireUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("lerobot.async_inference.helpers", "RemotePolicyConfig"),
        ("lerobot.async_inference.helpers", "TimedObservation"),
        ("lerobot.configs.types", "PolicyFeature"),
        ("lerobot.configs.types", "FeatureType"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        *((f"numpy.{core}.multiarray", name)
          for core in ("core", "_core") for name in ("_reconstruct", "scalar")),
        *((f"numpy.{core}.numeric", "_frombuffer") for core in ("core", "_core")),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")
        return super().find_class(module, name)


def _loads(data):
    return _WireUnpickler(io.BytesIO(data)).load()


def read_config(model_dir):
    """<model-dir>/config.txt, written by the converter.

    Split on the first space: line 1 is `instruction <free text>`. Repeated keys
    (cam0..N, rename) accumulate into lists.
    """
    cfg = {"cams": []}
    try:
        with open(os.path.join(model_dir, "config.txt")) as f:
            for line in f:
                k, _, v = line.partition(" ")
                v = v.strip()
                if k.startswith("cam") and k[3:].isdigit():
                    cfg["cams"].append(v)
                elif k == "rename":
                    cfg.setdefault("rename", []).append(v)
                else:
                    cfg[k] = v
    except OSError:
        pass
    return cfg


def _cam_list(args):
    """--cams as a list; Octo's converter records no camera names."""
    raw = getattr(args, "cams", None)
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else []


def _open_lib(lib_path, target):
    lib_path = os.path.expanduser(lib_path)
    if not os.path.exists(lib_path):
        sys.exit(
            f"No shared library at {lib_path}\n"
            f"  cmake --build build -j --target {target}\n"
            "or point --lib at it."
        )
    lib = ctypes.CDLL(lib_path)
    abi = getattr(lib, "vla_abi_version", None)
    if abi is not None:
        abi.restype = ctypes.c_int32
        abi = abi()
    if abi != VLA_ABI_VERSION:
        sys.exit(f"{lib_path} has ABI version {abi}, this server speaks {VLA_ABI_VERSION}: "
                 f"rebuild it (cmake --build build -j --target {target}) or update the server.")
    return lib


# ---------------------------------------------------------------------------
# engines: per-model, because the C ABIs differ
# ---------------------------------------------------------------------------
class _Engine:
    def __init__(self, args, name, converter, ints, predict_argtypes, with_tok=False):
        self.lib = _open_lib(args.lib, f"vla_simd_{name}")
        self.name = name
        self.model_dir = os.path.expanduser(args.model_dir)
        dirs = [self.model_dir]
        if with_tok:
            dirs.append(os.path.expanduser(args.tok_dir or os.path.join(self.model_dir, "tok")))

        self._fn("load").argtypes = [ctypes.c_char_p] * len(dirs)
        self._fn("load").restype = ctypes.c_void_p
        self._fn("free").argtypes = [ctypes.c_void_p]
        for i in ints:
            self._fn(i).argtypes = [ctypes.c_void_p]
            self._fn(i).restype = ctypes.c_int32
        self._fn("predict").argtypes = [ctypes.c_void_p, *predict_argtypes]
        self._fn("predict").restype = ctypes.c_int32

        self.h = self._fn("load")(*(d.encode() for d in dirs))
        if not self.h:
            sys.exit(f"vla_{name}_load failed for {' + '.join(dirs)} (run {converter})")
        for i in ints:
            setattr(self, i, self._fn(i)(self.h))

        self.cfg = read_config(self.model_dir)
        self.cam_names = self.cfg.get("cams", [])
        # The model owns mutable scratch and is not reentrant; gRPC serves on a pool.
        self.lock = threading.Lock()

    def _fn(self, suffix):
        return getattr(self.lib, f"vla_{self.name}_{suffix}")

    def _task(self, task):
        task = task or self.task
        if not task:
            raise RuntimeError("no task: the client sent none and the checkpoint recorded none")
        return task

    def _state(self, state):
        state = np.ascontiguousarray(np.asarray(state, np.float32).reshape(-1))
        if state.size != self.state_dim:
            raise ValueError(f"state has {state.size} values, the checkpoint wants {self.state_dim}")
        return state

    @staticmethod
    def _frames(frames, want):
        frames = np.ascontiguousarray(frames, np.uint8)
        if frames.shape != want:
            raise ValueError(f"frames {frames.shape} != {want}")
        return frames

    def _run(self, *argv, after=(), fn="predict"):
        out = np.empty((self.chunk, self.action_dim), np.float32)
        with self.lock:
            rc = self._fn(fn)(self.h, *argv, out.ctypes.data_as(F32P), *after)
        if rc != 0:
            raise RuntimeError(f"vla_{self.name}_{fn} failed (rc={rc})")
        return out

    def close(self):
        with self.lock:
            if self.h:
                self._fn("free")(self.h)
                self.h = None


class ActEngine(_Engine):
    """ctypes binding for libvla_simd_act.so (src/models/act/act_capi.cpp).

    The engine owns the whole processor pipeline: image rescale + per-channel
    normalization, state normalization and action un-normalization all happen
    inside predict(), from the stats baked into the converted checkpoint. Frames
    go in as raw uint8 at the checkpoint's resolution and actions come out in the
    robot's own units.
    """

    def __init__(self, args):
        super().__init__(args, "act", "tools/convert_act.py",
                         ("chunk", "action_dim", "state_dim", "n_cams", "img_h", "img_w"),
                         [U8P, F32P, ctypes.c_int32, F32P])
        self.n_views = self.n_cams

    def describe(self):
        return (f"{self.n_cams} cams {self.cam_names or '(unnamed)'} at "
                f"{self.img_h}x{self.img_w} | chunk {self.chunk}x{self.action_dim} "
                f"| state {self.state_dim}")

    def predict(self, adapted, seed):
        """frames [n_cams, H, W, 3] uint8, state [state_dim] -> [chunk, action_dim]."""
        state = self._state(adapted[1])
        frames = self._frames(adapted[0], (self.n_cams, self.img_h, self.img_w, 3))
        return self._run(frames.ctypes.data_as(U8P), state.ctypes.data_as(F32P),
                         1)  # un-normalize: actions come back in robot units

    def warmup_input(self):
        return (np.zeros((self.n_cams, self.img_h, self.img_w, 3), np.uint8),
                np.zeros(self.state_dim, np.float32))


class ImpactEngine(_Engine):
    """ctypes binding for libvla_simd_impact.so (src/models/impact/impact_capi.cpp).

    ACT's contract plus an instruction. As with ACT the engine owns the whole
    processor pipeline -- image rescale and per-channel normalization, state
    normalization, action un-normalization -- from the stats baked into the
    converted checkpoint, so raw uint8 frames go in and robot-unit actions come out.

    The instruction is passed on every call but only re-encoded when the string
    changes, so a rollout repeating one task pays for the T5-small tower and the
    FiLM head once per episode rather than once per query.
    """

    def __init__(self, args):
        super().__init__(args, "impact", "tools/convert_impact.py",
                         ("chunk", "action_dim", "state_dim", "n_cams", "img_h", "img_w", "n_text"),
                         [U8P, F32P, ctypes.c_char_p, ctypes.c_int32, F32P])
        self.n_views = self.n_cams
        self.task = None   # set by main() from --task or config.txt

    def describe(self):
        return (f"{self.n_cams} cams {self.cam_names or '(unnamed)'} at "
                f"{self.img_h}x{self.img_w} | chunk {self.chunk}x{self.action_dim} "
                f"| state {self.state_dim} | text {self.n_text}")

    def predict(self, adapted, seed):
        """frames [n_cams, H, W, 3] uint8, state, task -> [chunk, action_dim]."""
        task = self._task(adapted[2])
        state = self._state(adapted[1])
        frames = self._frames(adapted[0], (self.n_cams, self.img_h, self.img_w, 3))
        return self._run(frames.ctypes.data_as(U8P), state.ctypes.data_as(F32P), task.encode(),
                         1)  # un-normalize: actions come back in robot units

    def warmup_input(self):
        return (np.zeros((self.n_cams, self.img_h, self.img_w, 3), np.uint8),
                np.zeros(self.state_dim, np.float32), self.task)


class SmolvlaEngine(_Engine):
    """ctypes binding for libvla_simd_smolvla.so (src/models/smolvla/smolvla_capi.cpp).

    The engine owns the whole preprocessing pipeline: resize-with-pad, the
    [0,1]->[-1,1] map, the SmolVLM2 tokenizer, state normalization and action
    un-normalization all happen inside predict(), from the statistics baked into
    the converted checkpoint. Native-resolution frames go in and actions come out
    in the robot's own units.
    """

    def __init__(self, args):
        super().__init__(args, "smolvla", "tools/convert_hf_safetensors.py",
                         ("chunk", "action_dim", "state_dim", "n_views", "img_size", "tok_maxlen"),
                         [U8P, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                          I32P, I32P, ctypes.c_int32, F32P, F32P, ctypes.c_uint64, F32P],
                         with_tok=True)
        self.lib.vla_smolvla_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, I32P, I32P]
        self.lib.vla_smolvla_tokenize.restype = ctypes.c_int32
        self.task = None   # set by main() from --task or config.txt
        self._tokens = {}
        self.rtc_horizon, self.rtc_max_guidance = args.rtc_horizon, args.rtc_max_guidance
        self.rtc_delay, self.fps = args.rtc_delay, args.fps
        if self.rtc_horizon:
            self._fn("predict_rtc").argtypes = [*self._fn("predict").argtypes,
                                                F32P, ctypes.c_int32, F32P, ctypes.c_float]
            self._fn("predict_rtc").restype = ctypes.c_int32
        self._rtc_lock = threading.Lock()
        self.rtc_gen = 0
        self.reset_rtc()

    def reset_rtc(self):
        with self._rtc_lock:
            self._prev = None
            self._lat = deque(maxlen=5)
            self.rtc_gen += 1

    def _guide(self, timestep):
        w = np.zeros(self.chunk, np.float32)
        if timestep is None or self._prev is None or not 0 <= timestep - self._prev[0] < len(self._prev[1]):
            return np.zeros((0, self.action_dim), np.float32), w
        left = np.ascontiguousarray(self._prev[1][timestep - self._prev[0]:])
        d = self.rtc_delay if self.rtc_delay is not None else 1 + math.ceil(max(self._lat) * self.fps)
        h = min(self.rtc_horizon, len(left))
        s = min(d, h)
        w[:s] = 1
        w[s:h] = np.linspace(1, 0, h - s + 2)[1:-1]
        return left, w

    def describe(self):
        return (f"{self.n_views} views {self.cam_names or '(unnamed)'} -> {self.img_size}px "
                f"| chunk {self.chunk}x{self.action_dim} | state {self.state_dim} "
                f"| steps {self.cfg.get('num_steps', '?')}")

    TOKEN_CACHE_MAX = 64   # the task arrives per observation; a varying one must not grow this

    def tokenize(self, task):
        """Cached ids/mask. The engine appends lerobot's trailing newline itself."""
        if task not in self._tokens:
            if len(self._tokens) >= self.TOKEN_CACHE_MAX:
                self._tokens.clear()
            ids = np.zeros(self.tok_maxlen, np.int32)
            mask = np.zeros(self.tok_maxlen, np.int32)
            with self.lock:   # tokenize enters the same handle as predict
                n = self.lib.vla_smolvla_tokenize(
                    self.h, task.encode(),
                    ids.ctypes.data_as(I32P), mask.ctypes.data_as(I32P))
            if n < 0:
                raise RuntimeError("vla_smolvla_tokenize failed")
            logger.info("task %r -> %d tokens %s", task, n, ids[:n].tolist())
            self._tokens[task] = (ids, mask)
        return self._tokens[task]

    def predict(self, adapted, seed, rtc=None):
        """frames [n_views, H, W, 3] uint8 native res -> [chunk, action_dim]."""
        frames, state, task = adapted
        task = self._task(task)
        frames = np.ascontiguousarray(frames, np.uint8)
        state = self._state(state)
        if frames.ndim != 4 or frames.shape[0] != self.n_views or frames.shape[3] != 3:
            raise ValueError(f"frames {frames.shape} != ({self.n_views}, H, W, 3)")
        if frames.shape[1] <= 0 or frames.shape[2] <= 0:
            raise ValueError(f"frames have a zero dimension: {frames.shape}")
        ids, mask = self.tokenize(task)
        argv = (frames.ctypes.data_as(U8P), self.n_views,
                frames.shape[1], frames.shape[2],
                ids.ctypes.data_as(I32P), mask.ctypes.data_as(I32P), self.tok_maxlen,
                state.ctypes.data_as(F32P), None, seed)
        if rtc is None:
            return self._run(*argv)
        timestep, keep, since, gen = rtc
        with self._rtc_lock:
            live = gen == self.rtc_gen
            left, w = self._guide(timestep if live else None)
            out = self._run(*argv, fn="predict_rtc",
                            after=(left.ctypes.data_as(F32P), len(left), w.ctypes.data_as(F32P),
                                   self.rtc_max_guidance))
            if live:
                self._lat.append(time.perf_counter() - since)
                self._prev = (timestep, out[:keep])
        return out

    def warmup_input(self):
        return (np.zeros((self.n_views, 480, 640, 3), np.uint8),
                np.zeros(self.state_dim, np.float32), self.task)


class OctoEngine(_Engine):
    """ctypes binding for libvla_simd_octo.so (src/models/octo/octo_capi.cpp).

    Two differences from every other engine here. Octo takes no proprioceptive
    state - it is vision plus language only - and it consumes an observation
    *window*, so predict() is handed `wnd` frames per camera plus a mask marking
    which of them are real rather than start-of-episode padding.

    The two towers also run at fixed, different resolutions (256 primary, 128
    wrist) and the engine does not resize, so the adapter does.
    """

    PRIMARY_HW = (256, 256)
    WRIST_HW = (128, 128)

    def __init__(self, args):
        super().__init__(args, "octo", "tools/convert_octo.py",
                         ("horizon", "action_dim", "max_window"),
                         [U8P, U8P, ctypes.c_int32, U8P, ctypes.c_char_p,
                          ctypes.c_uint64, ctypes.c_int32, F32P],
                         with_tok=True)
        self.chunk = self.horizon
        self.window = min(int(self.cfg.get("window", 2) or 2), self.max_window)
        self.cam_names = self.cam_names or _cam_list(args)
        self.n_views = 1 if len(self.cam_names) == 1 else 2
        self.task = None

    def describe(self):
        return (f"{self.n_views} cams {self.cam_names or '(unnamed)'} at 256/128 "
                f"| window {self.window}/{self.max_window} | chunk {self.chunk}x{self.action_dim} | no state "
                f"| steps {self.cfg.get('steps', '?')}")

    def predict(self, adapted, seed):
        """(primary [wnd,256,256,3], wrist [wnd,128,128,3], mask [wnd], task)."""
        primary, wrist, mask, task = adapted
        task = self._task(task)
        primary = np.ascontiguousarray(primary, np.uint8)
        wrist = None if wrist is None else np.ascontiguousarray(wrist, np.uint8)
        mask = np.ascontiguousarray(mask, np.uint8)
        wnd = primary.shape[0]
        wrist_wnd = wnd if wrist is None else wrist.shape[0]
        if wrist_wnd != wnd or mask.shape[0] != wnd:
            raise ValueError(f"window mismatch: primary {wnd}, wrist {wrist_wnd}, "
                             f"mask {mask.shape[0]}")
        if not 1 <= wnd <= self.max_window:
            raise ValueError(f"window {wnd} outside [1, {self.max_window}]")
        return self._run(primary.ctypes.data_as(U8P),
                         None if wrist is None else wrist.ctypes.data_as(U8P),
                         wnd, mask.ctypes.data_as(U8P), task.encode(), seed,
                         1)  # un-normalize: actions come back in dataset units

    def warmup_input(self):
        w = self.window
        return (np.zeros((w, *self.PRIMARY_HW, 3), np.uint8),
                np.zeros((w, *self.WRIST_HW, 3), np.uint8) if self.n_views == 2 else None,
                np.ones(w, np.uint8), self.task)


class TurboVlaEngine(_Engine):
    """ctypes binding for libvla_simd_turbovla.so (src/models/turbovla/turbovla_capi.cpp).

    IMPACT's shape - frames, state, instruction - but the frames are consumed as
    given: the reference policy does no resize and no rotation, so the adapter
    rejects a mismatched resolution instead of resampling into it.
    """

    def __init__(self, args):
        super().__init__(args, "turbovla", "tools/convert_turbovla.py",
                         ("chunk", "action_dim", "state_dim", "n_views", "img_size"),
                         [U8P, F32P, ctypes.c_char_p, ctypes.c_int32, F32P])
        self.task = None

    def describe(self):
        return (f"{self.n_views} views {self.cam_names or '(unnamed)'} at "
                f"{self.img_size}x{self.img_size} | chunk {self.chunk}x{self.action_dim} "
                f"| state {self.state_dim}")

    def predict(self, adapted, seed):
        """frames [n_views, img, img, 3] uint8, state, task -> [chunk, action_dim]."""
        frames, state, task = adapted
        task = self._task(task)
        state = self._state(state)
        frames = self._frames(frames, (self.n_views, self.img_size, self.img_size, 3))
        return self._run(frames.ctypes.data_as(U8P), state.ctypes.data_as(F32P), task.encode(),
                         1)  # un-normalize: actions come back in env units

    def warmup_input(self):
        return (np.zeros((self.n_views, self.img_size, self.img_size, 3), np.uint8),
                np.zeros(self.state_dim, np.float32), self.task)


class DiffusionEngine(_Engine):
    """ctypes binding for libvla_simd_diffusion.so (src/models/diffusion/diffusion_capi.cpp).

    Takes an observation history: n_obs_steps frames per camera and n_obs_steps
    states, oldest first. The sampler's noise is supplied by the caller rather
    than drawn inside the engine - passing NULL starts the reverse process from
    zeros, which is not a sample - so the server draws it here from the query
    seed, which also makes a rollout reproducible.
    """

    def __init__(self, args):
        super().__init__(args, "diffusion", "tools/convert_diffusion.py",
                         ("chunk", "horizon", "action_dim", "state_dim", "n_cams",
                          "n_obs_steps", "img_h", "img_w", "num_steps", "is_ddim"),
                         [U8P, F32P, ctypes.c_int32, F32P, F32P])
        self.is_ddim = bool(self.is_ddim)
        self.n_views = self.n_cams

    def describe(self):
        return (f"{self.n_cams} cams {self.cam_names or '(unnamed)'} at "
                f"{self.img_h}x{self.img_w} x {self.n_obs_steps} obs steps "
                f"| chunk {self.chunk}x{self.action_dim} (horizon {self.horizon}) "
                f"| state {self.state_dim} "
                f"| {'DDIM' if self.is_ddim else 'DDPM'} x{self.num_steps}")

    def predict(self, adapted, seed):
        """frames [n_obs, n_cams, H, W, 3] uint8, state [n_obs, state_dim]."""
        frames = self._frames(adapted[0], (self.n_obs_steps, self.n_cams, self.img_h, self.img_w, 3))
        state = np.ascontiguousarray(np.asarray(adapted[1], np.float32))
        if state.shape != (self.n_obs_steps, self.state_dim):
            raise ValueError(f"state {state.shape} != {(self.n_obs_steps, self.state_dim)}")

        # The prior plus one buffer per DDPM step; DDIM reads only the prior, but
        # the engine indexes the array either way, so it is always sized for the
        # scheduler that is actually loaded.
        rng = np.random.default_rng(seed)
        noise = np.ascontiguousarray(
            rng.standard_normal((1 + self.num_steps, self.horizon, self.action_dim),
                                dtype=np.float32))
        return self._run(frames.ctypes.data_as(U8P), state.ctypes.data_as(F32P),
                         1,  # un-normalize: actions come back in robot units
                         after=(noise.ctypes.data_as(F32P),))

    def warmup_input(self):
        return (np.zeros((self.n_obs_steps, self.n_cams, self.img_h, self.img_w, 3), np.uint8),
                np.zeros((self.n_obs_steps, self.state_dim), np.float32))


# ---------------------------------------------------------------------------
# observation mapping
# ---------------------------------------------------------------------------
class ObservationAdapter:
    """Raw robot observation -> the engine's input tuple.

    The client sends whatever the robot produced: `{"<motor>.pos": float, ...,
    "<camera>": HxWx3 uint8, "task": str}`. Turning that into a flat state vector
    and an ordered camera stack is exactly what lerobot's own feature mapping
    does, so this defers to `build_dataset_frame` rather than reimplementing the
    key convention.

    Camera order is the checkpoint's (config.txt), not the robot's dict order:
    the views are position-dependent and swapping front/wrist silently produces
    plausible, wrong actions.
    """

    def __init__(self, engine, lerobot_features):
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR

        self.engine = engine
        self.features = lerobot_features
        self.OBS_STR = OBS_STR
        self.OBS_STATE = OBS_STATE

        available = [k for k in lerobot_features if k.startswith(OBS_IMAGES)]
        if engine.cam_names:
            wanted = [f"{OBS_IMAGES}.{c}" for c in engine.cam_names]
            missing = [k for k in wanted if k not in available]
            if missing:
                # the converter records the checkpoint's names; the client may be
                # sending the dataset's pre-rename ones (front -> camera1)
                alias = {}
                for entry in engine.cfg.get("rename", []):
                    src, _, dst = entry.partition(" ")
                    alias[f"{OBS_IMAGES}.{dst}"] = f"{OBS_IMAGES}.{src}"
                wanted = [alias.get(k, k) for k in wanted]
                missing = [k for k in wanted if k not in available]
            if missing:
                raise ValueError(
                    f"the client offers cameras {sorted(available)}, the checkpoint needs {wanted} "
                    f"(missing {missing})"
                )
            self.camera_keys = wanted
        else:
            # a checkpoint converted without camera names: fall back to the
            # client's order and say so, since it may not be the training order
            self.camera_keys = sorted(available)
            logger.warning(
                "checkpoint recorded no camera names; using client order %s. "
                "Re-run the converter so config.txt pins the order.",
                self.camera_keys,
            )

        if len(self.camera_keys) != engine.n_views:
            raise ValueError(
                f"checkpoint wants {engine.n_views} views, client offers {len(self.camera_keys)}"
            )

    def _frame(self, raw_observation):
        from lerobot.utils.feature_utils import build_dataset_frame
        return build_dataset_frame(self.features, raw_observation, prefix=self.OBS_STR)

    @staticmethod
    def _as_uint8(img):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB frame, got {img.shape}")
        if img.dtype == np.uint8:
            return img
        # a client that already scaled to [0,1] floats
        scale = 255.0 if float(img.max()) <= 1.0 else 1.0
        return np.clip(img * scale, 0, 255).astype(np.uint8)

    def _resize(self, img, h, w):
        """uint8 HWC at exactly h x w, resampling only when it has to."""
        img = self._as_uint8(img)
        if img.shape[:2] == (h, w):
            return img

        if not getattr(self, "_warned_resize", False):
            logger.warning(
                "camera frames are %dx%d, the checkpoint expects %dx%d: resizing on the server. "
                "Configure the camera at the training resolution to avoid the resample.",
                img.shape[0], img.shape[1], h, w,
            )
            self._warned_resize = True

        # Bilinear, matching lerobot's resize_robot_observation_image, then back to
        # uint8 because that is what the engine's preprocessing consumes.
        import torch

        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0).float()
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


class ActAdapter(ObservationAdapter):
    """ACT consumes frames at the checkpoint's resolution, so this one resizes."""

    def __call__(self, raw_observation):
        frame = self._frame(raw_observation)
        state = np.asarray(frame[self.OBS_STATE], np.float32).reshape(-1)
        frames = np.empty(
            (self.engine.n_cams, self.engine.img_h, self.engine.img_w, 3), np.uint8)
        for i, key in enumerate(self.camera_keys):
            frames[i] = self._resize(np.asarray(frame[key]), self.engine.img_h, self.engine.img_w)
        return frames, state


class ImpactAdapter(ActAdapter):
    """ACT's resize, plus the task: IMPACT takes frames at the checkpoint's
    resolution and needs the instruction."""

    def __call__(self, raw_observation):
        frames, state = super().__call__(raw_observation)
        return frames, state, raw_observation.get("task")


class SmolvlaAdapter(ObservationAdapter):
    """SmolVLA's resize-with-pad is part of the model and lives in the engine, so
    frames cross the wire at the camera's own resolution. Also carries the task."""

    def __call__(self, raw_observation):
        frame = self._frame(raw_observation)
        state = np.asarray(frame[self.OBS_STATE], np.float32).reshape(-1)
        frames = [self._as_uint8(np.asarray(frame[key])) for key in self.camera_keys]
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise ValueError(f"all views must share one resolution, got {shapes}")
        return np.stack(frames), state, raw_observation.get("task")


class History:
    """The last n observations, oldest first, for the policies that want a window.

    lerobot's async protocol carries one observation per message, so the window
    is assembled here. Before n have arrived the earliest one is repeated, which
    is what the reference policies do at the start of an episode, and the mask
    says which entries are real.
    """

    def __init__(self, n):
        self.n = n
        self.items = []
        self.lock = threading.Lock()

    def push(self, item):
        """(window, mask): the buffer left-padded to n by repeating the oldest."""
        with self.lock:
            self.items.append(item)
            if len(self.items) > self.n:
                self.items.pop(0)
            k = len(self.items)
            pad = self.n - k
            window = [self.items[0]] * pad + self.items
        return window, np.array([0] * pad + [1] * k, np.uint8)


class TurboVlaAdapter(ObservationAdapter):
    """Frames at the checkpoint's square resolution, plus the task.

    Unlike ACT this one does not resample: the reference policy consumes its
    frames as given, so a wrong resolution is a configuration error rather than
    something to paper over.
    """

    def __call__(self, raw_observation):
        frame = self._frame(raw_observation)
        state = np.asarray(frame[self.OBS_STATE], np.float32).reshape(-1)
        size = self.engine.img_size
        frames = np.empty((self.engine.n_views, size, size, 3), np.uint8)
        for i, key in enumerate(self.camera_keys):
            img = self._as_uint8(np.asarray(frame[key]))
            if img.shape[:2] != (size, size):
                raise ValueError(
                    f"camera {key} is {img.shape[0]}x{img.shape[1]}, TurboVLA consumes frames as "
                    f"given and wants {size}x{size}; resize at the source"
                )
            frames[i] = img
        return frames, state, raw_observation.get("task")


class OctoAdapter(ObservationAdapter):
    """Two towers at fixed, different resolutions, a window of frames, no state."""

    def __init__(self, engine, lerobot_features):
        super().__init__(engine, lerobot_features)
        self.history = History(engine.window)

    def __call__(self, raw_observation):
        frame = self._frame(raw_observation)
        primary = self._resize(np.asarray(frame[self.camera_keys[0]]), *self.engine.PRIMARY_HW)
        wrist = (self._resize(np.asarray(frame[self.camera_keys[1]]), *self.engine.WRIST_HW)
                 if self.engine.n_views == 2 else None)
        window, mask = self.history.push((primary, wrist))
        return (np.stack([p for p, _ in window]),
                None if wrist is None else np.stack([w for _, w in window]),
                mask,
                raw_observation.get("task"))


class DiffusionAdapter(ActAdapter):
    """ACT's per-camera resize, over a window of observations rather than one."""

    def __init__(self, engine, lerobot_features):
        super().__init__(engine, lerobot_features)
        self.history = History(engine.n_obs_steps)

    def __call__(self, raw_observation):
        frames, state = super().__call__(raw_observation)
        window, _ = self.history.push((frames, state))
        return (np.stack([f for f, _ in window]), np.stack([s for _, s in window]))


# ---------------------------------------------------------------------------
# model specs: everything the shared server needs to know about a model
# ---------------------------------------------------------------------------
class ModelSpec:
    def __init__(self, name, policy_type, engine_cls, adapter_cls, lib,
                 omp_threads, obs_queue_timeout, extra_args=()):
        self.name = name
        self.policy_type = policy_type
        self.engine_cls = engine_cls
        self.adapter_cls = adapter_cls
        paths = [os.path.join(d, lib + LIB_EXT) for d in LIB_DIRS]
        self.default_lib = next((p for p in paths if os.path.exists(p)), paths[-1])
        self.omp_threads = omp_threads
        self.obs_queue_timeout = obs_queue_timeout
        self.extra_args = extra_args


def _smolvla_extra(p):
    p.add_argument("--tok-dir", default=None, help="default: <model-dir>/tok")
    p.add_argument("--task", default=None,
                   help="instruction; default is the one the converter recorded")
    p.add_argument("--seed", type=int, default=0,
                   help="flow-matching noise for query i is seed+i (default: 0)")
    p.add_argument("--rtc-horizon", type=int, default=0,
                   help="real-time chunking execution horizon in actions; 0 = off (default: 0)")
    p.add_argument("--rtc-max-guidance", type=float, default=10.0,
                   help="RTC guidance weight cap (default: 10.0)")
    p.add_argument("--rtc-delay", type=int, default=None,
                   help="RTC inference delay in actions (default: measured from latency and --fps)")


ACT = ModelSpec("ACT", "act", ActEngine, ActAdapter, "libvla_simd_act",
                omp_threads="6", obs_queue_timeout=2.0)
SMOLVLA = ModelSpec("SmolVLA", "smolvla", SmolvlaEngine, SmolvlaAdapter,
                    "libvla_simd_smolvla",
                    omp_threads="4", obs_queue_timeout=5.0, extra_args=(_smolvla_extra,))


def _lang_extra(p):
    p.add_argument("--task", default=None,
                   help="instruction; default is the one the converter recorded")


IMPACT = ModelSpec("IMPACT", "impact", ImpactEngine, ImpactAdapter,
                   "libvla_simd_impact",
                   omp_threads="6", obs_queue_timeout=2.0, extra_args=(_lang_extra,))


def _octo_extra(p):
    p.add_argument("--tok-dir", default=None, help="default: <model-dir>/tok")
    _lang_extra(p)
    p.add_argument("--cams", default=None,
                   help="camera order, primary first (e.g. 'primary,wrist', or 'primary' "
                        "alone for no wrist); Octo's converter records no names and the "
                        "two towers differ in size")
    p.add_argument("--seed", type=int, default=0,
                   help="DDPM noise for query i is seed+i (default: 0)")


def _diffusion_extra(p):
    p.add_argument("--seed", type=int, default=0,
                   help="sampler noise for query i is seed+i (default: 0)")


OCTO = ModelSpec("Octo-Small", "octo", OctoEngine, OctoAdapter,
                 "libvla_simd_octo",
                 omp_threads="8", obs_queue_timeout=2.0, extra_args=(_octo_extra,))
TURBOVLA = ModelSpec("TurboVLA", "turbovla", TurboVlaEngine, TurboVlaAdapter,
                     "libvla_simd_turbovla",
                     omp_threads="8", obs_queue_timeout=2.0, extra_args=(_lang_extra,))
DIFFUSION = ModelSpec("Diffusion Policy", "diffusion", DiffusionEngine, DiffusionAdapter,
                      "libvla_simd_diffusion",
                      omp_threads="6", obs_queue_timeout=5.0, extra_args=(_diffusion_extra,))

MODELS = {
    "act": ACT,
    "diffusion": DIFFUSION,
    "impact": IMPACT,
    "octo": OCTO,
    "smolvla": SMOLVLA,
    "turbovla": TURBOVLA,
}


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
def build_servicer_class(spec):
    """Imported lazily so `--help` works without lerobot on the path."""
    import torch
    from lerobot.async_inference.helpers import (
        RemotePolicyConfig,
        TimedAction,
        TimedObservation,
        observations_similar,
    )
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks

    class ActionPickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, torch.Tensor):
                return torch.from_numpy, (obj.numpy(),)
            return NotImplemented

    class VlaSimdPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
        """lerobot's AsyncInference service, backed by the vla.simd engine.

        The queueing behaviour mirrors the reference server: a depth-1 queue that
        always holds the freshest observation, plus the must-go / already-predicted
        / too-similar filters, so a client tuned against the reference server sees
        the same dynamics here.
        """

        def __init__(self, engine, cfg):
            self.engine = engine
            self.cfg = cfg
            self.observation_queue = Queue(maxsize=1)
            self.shutdown_event = threading.Event()

            self._predicted_lock = threading.Lock()
            self._predicted_timesteps = set()
            self.last_processed_obs = None

            self.adapter = None
            self.lerobot_features = None
            self.actions_per_chunk = engine.chunk
            self._query_lock = threading.Lock()
            self.n_queries = 0

        def _reset(self):
            self.shutdown_event.set()
            self.observation_queue = Queue(maxsize=1)
            with self._predicted_lock:
                self._predicted_timesteps = set()
            self.last_processed_obs = None
            # A reconnecting client that never re-sends instructions must not
            # inherit the previous client's feature mapping and camera keys.
            self.adapter = None
            self.lerobot_features = None
            if self.cfg.rtc_horizon:
                self.engine.reset_rtc()

        # -- rpc -------------------------------------------------------------
        def Ready(self, request, context):  # noqa: N802
            logger.info("client %s ready", context.peer())
            self._reset()
            self.shutdown_event.clear()
            return services_pb2.Empty()

        def SendPolicyInstructions(self, request, context):  # noqa: N802
            try:
                specs = _loads(request.data)
                if not isinstance(specs, RemotePolicyConfig):
                    raise TypeError(f"expected a RemotePolicyConfig, got {type(specs)}")
                if specs.policy_type != spec.policy_type:
                    raise ValueError(
                        f"this server serves {spec.policy_type} only, "
                        f"the client asked for {specs.policy_type!r}"
                    )
                if not isinstance(specs.actions_per_chunk, int) or specs.actions_per_chunk < 1:
                    raise ValueError(
                        f"actions_per_chunk must be an int >= 1, got {specs.actions_per_chunk!r}")

                # The checkpoint is the converted one in --model-dir; a mismatch
                # here is the classic "served the wrong model" bug, so it is loud.
                if self.cfg.checkpoint and specs.pretrained_name_or_path != self.cfg.checkpoint:
                    logger.warning(
                        "client asked for %r but this server serves %r from %s",
                        specs.pretrained_name_or_path, self.cfg.checkpoint, self.cfg.model_dir,
                    )

                self.lerobot_features = specs.lerobot_features
                self.adapter = spec.adapter_cls(self.engine, specs.lerobot_features)
                self.actions_per_chunk = min(specs.actions_per_chunk, self.engine.chunk)
            except Exception as e:
                # a bare raise inside a handler reaches the client as UNKNOWN
                logger.exception("rejecting policy instructions from %s", context.peer())
                context.abort(_grpc_status().INVALID_ARGUMENT, str(e))

            logger.info(
                "policy instructions from %s | actions_per_chunk %d (chunk %d) | cameras %s",
                context.peer(), self.actions_per_chunk, self.engine.chunk, self.adapter.camera_keys,
            )
            return services_pb2.Empty()

        def SendObservations(self, request_iterator, context):  # noqa: N802
            received = receive_bytes_in_chunks(
                request_iterator, None, self.shutdown_event, f"{spec.policy_type}_policy_server")
            adapter = self.adapter
            try:
                obs = _loads(received)
                if not isinstance(obs, TimedObservation):
                    raise TypeError(f"expected a TimedObservation, got {type(obs)}")
                adapted = adapter(obs.get_observation()) if hasattr(adapter, "history") else None
            except Exception as e:
                logger.exception("bad observation from %s", context.peer())
                context.abort(_grpc_status().INVALID_ARGUMENT, f"bad observation: {e}")
            if not self._enqueue(obs, adapted):
                logger.debug("observation #%s filtered out", obs.get_timestep())
            return services_pb2.Empty()

        def GetActions(self, request, context):  # noqa: N802
            try:
                obs, adapted, stamp = self.observation_queue.get(timeout=self.cfg.obs_queue_timeout)
            except Empty:
                return services_pb2.Empty()

            try:
                with self._predicted_lock:
                    self._predicted_timesteps.add(obs.get_timestep())

                t0 = time.perf_counter()
                chunk = self._predict(obs, adapted, stamp)
                inference_ms = (time.perf_counter() - t0) * 1000

                with self._query_lock:
                    n = self.n_queries
                if n % 10 == 1:
                    logger.info(
                        "chunk #%s | %d actions | %.0f ms | action[0]=%s",
                        obs.get_timestep(), len(chunk), inference_ms,
                        np.round(np.asarray(chunk[0].get_action()), 3).tolist(),
                    )
                buf = io.BytesIO()
                ActionPickler(buf).dump(chunk)
                return services_pb2.Actions(data=buf.getvalue())
            except Exception as e:
                # An empty reply is what a timed-out queue returns, so a broken
                # engine would be indistinguishable from a quiet client.
                logger.exception("inference failed for observation #%s", obs.get_timestep())
                context.abort(_grpc_status().INTERNAL, f"inference failed: {e}")

        # -- internals -------------------------------------------------------
        def _sanity_ok(self, obs, previous):
            with self._predicted_lock:
                already = obs.get_timestep() in self._predicted_timesteps
            if already:
                return False
            return not observations_similar(obs, previous, lerobot_features=self.lerobot_features)

        def _enqueue(self, obs, adapted):
            if not (obs.must_go or self.last_processed_obs is None
                    or self._sanity_ok(obs, self.last_processed_obs)):
                return False
            # Drop-oldest on a maxsize=1 queue. Both calls are non-blocking:
            # another worker may drain between the two, and a blocking put would
            # then park this gRPC worker forever.
            try:
                self.observation_queue.get_nowait()
            except Empty:
                pass
            try:
                self.observation_queue.put_nowait(
                    (obs, adapted, (time.perf_counter(), self.engine.rtc_gen) if self.cfg.rtc_horizon else None))
            except Full:
                return False
            return True

        def _predict(self, timed_obs: "TimedObservation", adapted, stamp=None):
            if self.adapter is None:
                raise RuntimeError("no policy instructions received yet")

            if adapted is None:
                adapted = self.adapter(timed_obs.get_observation())
            self.last_processed_obs = timed_obs

            # Reserve AND advance under one lock. Incrementing only after
            # _predict returned let two concurrent workers take the same index,
            # so two chunks were sampled from the same flow-matching seed.
            with self._query_lock:
                index = self.n_queries
                self.n_queries += 1
            kw = {"rtc": (timed_obs.get_timestep(), self.actions_per_chunk, *stamp)} if stamp else {}
            actions = self.engine.predict(adapted, self.cfg.seed + index, **kw)[: self.actions_per_chunk]

            t0 = timed_obs.get_timestamp()
            i0 = timed_obs.get_timestep()
            return [
                TimedAction(
                    timestamp=t0 + i * self.cfg.environment_dt,
                    timestep=i0 + i,
                    action=torch.from_numpy(np.ascontiguousarray(a)),
                )
                for i, a in enumerate(actions)
            ]

    return VlaSimdPolicyServer, services_pb2_grpc


def _grpc_status():
    import grpc
    return grpc.StatusCode


class ServerConfig:
    def __init__(self, args):
        self.model_dir = args.model_dir
        self.checkpoint = args.checkpoint
        self.obs_queue_timeout = args.obs_queue_timeout
        self.environment_dt = 1.0 / args.fps
        self.seed = getattr(args, "seed", 0)
        self.rtc_horizon = getattr(args, "rtc_horizon", 0)


def main():
    # Resolved before the real parser is built: the model decides the
    # defaults, the extra flags and the OMP thread count below.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=sorted(MODELS))
    known, _ = pre.parse_known_args()
    if not known.model:
        sys.exit(
            "pass --model: " + ", ".join(sorted(MODELS)) + "\n"
            "  vla-simd-serve --model impact --model-dir "
            "hf://khanhnd61/impact-so101-multi-task-gguf/impact-so101-multi-task.gguf"
        )
    spec = MODELS[known.model]

    # libgomp reads this when the shared library loads, so set it before the
    # first import that pulls it in. OMP_PROC_BIND/OMP_PLACES are deliberately
    # not set: pinning to `cores` measured ~3x slower on a hybrid P+E-core part,
    # where a contiguous run of "cores" mixes the two classes and every barrier
    # waits on the slowest. Set them yourself on a homogeneous machine.
    # The per-model default is tuned for a desktop part; cap it at the machine's
    # cores so serving on a 4-core Pi does not silently oversubscribe (which costs
    # ~6% and shows up in the log as a thread count the board does not have). An
    # explicit OMP_NUM_THREADS still wins - thread count is a reported parameter.
    ncpu = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(min(int(spec.omp_threads), ncpu)))

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), default=spec.policy_type,
                   help="which policy to serve")
    p.add_argument("--model-dir", required=True,
                   help="a .gguf, a directory holding one, or hf://<user>/<repo>[@<revision>]"
                        "[/<file>.gguf] on the Hugging Face Hub")
    p.add_argument("--lib", default=spec.default_lib,
                   help=f"path to {os.path.basename(spec.default_lib)}")
    # Loopback by default: the payload codec is pickle over an unauthenticated
    # port, in the control path of a physical arm. Exposing it to the LAN is an
    # explicit choice.
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 keeps it local; 0.0.0.0 serves the LAN (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fps", type=int, default=30,
                   help="control rate the action timestamps are spaced at (default: 30)")
    p.add_argument("--obs-queue-timeout", type=float, default=spec.obs_queue_timeout)
    p.add_argument("--checkpoint", default=None,
                   help="hub id this model-dir was converted from; warns on a client mismatch")
    p.add_argument("--workers", type=int, default=4, help="gRPC thread pool size (default: 4)")
    p.add_argument("--max-message-mb", type=int, default=64,
                   help="gRPC receive/send cap; grpc's own default is 4 MB (default: 64)")
    p.add_argument("--int8", type=int, default=None, metavar="MASK",
                   help="W8A8 layer-group bitmask, exported as <MODEL>_INT8 (act, diffusion, impact, octo, smolvla)")
    p.add_argument("--bench", type=int, default=0, metavar="N",
                   help="time N queries after warmup, print the latency and exit")
    p.add_argument("--soak", type=float, default=0.0, metavar="SEC",
                   help="like --bench, but keep querying for SEC seconds")
    p.add_argument("--json", action="store_true", help="print the --bench/--soak result as JSON")
    p.add_argument("--selftest", action="store_true",
                   help="load, run one dummy query, print the latency and exit")
    for add in spec.extra_args:
        add(p)
    args = p.parse_args()

    # `seed` only exists for the models whose spec adds it, so check what is there
    # rather than assuming every model took every optional flag.
    for name, lo in (("fps", 1), ("port", 1), ("workers", 1), ("max_message_mb", 1), ("seed", 0),
                     ("bench", 0), ("soak", 0), ("rtc_horizon", 0), ("rtc_delay", 0)):
        if getattr(args, name, None) is None:
            continue
        if getattr(args, name) < lo:
            p.error(f"--{name.replace('_', '-')} must be >= {lo}")
    if args.obs_queue_timeout <= 0:
        p.error("--obs-queue-timeout must be > 0")
    if not 0 < getattr(args, "rtc_max_guidance", 1.0) < math.inf:
        p.error("--rtc-max-guidance must be > 0 and finite")

    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    if args.int8 is not None:
        if spec.policy_type not in ("act", "diffusion", "impact", "octo", "smolvla"):
            p.error(f"--int8 is not implemented for {spec.policy_type}")
        os.environ[f"{spec.policy_type.upper()}_INT8"] = str(args.int8)

    if args.model_dir.startswith("hf://"):
        # hf://<user>/<repo>[@<revision>][/<file>.gguf]: a file picks one GGUF of
        # several in the repo, and only that file is downloaded
        ref, _, rev = args.model_dir[5:].partition("@")
        rev, _, rev_file = rev.partition("/")
        parts = ref.split("/")
        repo, file = "/".join(parts[:2]), "/".join(parts[2:]) or rev_file
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            args.model_dir = (hf_hub_download(repo, file, revision=rev or None) if file
                              else snapshot_download(repo, revision=rev or None))
        except Exception as e:
            sys.exit(f"--model-dir {args.model_dir}: Hub download failed ({e})")
    # a vla.cpp GGUF loads as is; stage the tokenizer/statistics it does not carry
    if __package__:
        from . import gguf_stage
    else:
        import gguf_stage
    args.model_dir = gguf_stage.stage(args.model_dir, spec.policy_type)

    t0 = time.time()
    engine = spec.engine_cls(args)
    # Every language-conditioned engine carries a default task; ACT does not have
    # the attribute at all. Keyed on the attribute rather than on a class so a new
    # model gets this by declaring `self.task`, not by being added to an isinstance.
    if hasattr(engine, "task"):
        engine.task = getattr(args, "task", None) or engine.cfg.get("instruction")
        if not engine.task:
            engine.close()
            sys.exit(f"No instruction in {args.model_dir}/config.txt - pass --task.")
    engine.lib.vla_backend_name.restype = ctypes.c_char_p
    engine.lib.vla_int8_available.restype = ctypes.c_int32
    backend = engine.lib.vla_backend_name().decode()
    int8 = bool(engine.lib.vla_int8_available())
    logger.info("loaded %s in %.1f s | %s | backend %s | int8 %s | threads %s",
                args.model_dir, time.time() - t0, engine.describe(), backend,
                "yes" if int8 else "no", os.environ["OMP_NUM_THREADS"])

    try:
        # Warm up before the first client: the first predict pays for every
        # scratch allocation and a cold weight arena.
        engine.predict(engine.warmup_input(), 0)
        t0 = time.time()
        engine.predict(engine.warmup_input(), 0)
        logger.info("warm inference: %.0f ms per chunk", (time.time() - t0) * 1000)
        if args.selftest:
            return
        if args.bench > 0 or args.soak > 0:
            rng = np.random.default_rng(0)
            inp = [a.copy() if isinstance(a, np.ndarray) else a for a in engine.warmup_input()]
            ts = []
            t_end = time.perf_counter() + args.soak
            while len(ts) < args.bench or time.perf_counter() < t_end:
                for a in inp:
                    if isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.ndim >= 3:
                        a[...] = rng.integers(0, 256, a.shape, np.uint8)
                t0 = time.perf_counter()
                engine.predict(tuple(inp), len(ts))
                ts.append((time.perf_counter() - t0) * 1000)
            p10, med, p90 = np.percentile(ts, [10, 50, 90])
            res = {"model": spec.policy_type, "backend": backend, "int8": args.int8 or 0,
                   "threads": int(os.environ["OMP_NUM_THREADS"]), "queries": len(ts),
                   "median_ms": round(float(med), 2), "p10_ms": round(float(p10), 2),
                   "p90_ms": round(float(p90), 2),
                   "actions_per_s": round(engine.chunk * 1000 / float(med), 1)}
            if args.json:
                print(json.dumps(res))
            else:
                print(f"{spec.name} on {backend} x{res['threads']}: median {res['median_ms']} ms "
                      f"(p10 {res['p10_ms']}, p90 {res['p90_ms']}) over {len(ts)} queries, "
                      f"{res['actions_per_s']} actions/s")
            return

        try:
            import grpc
        except ImportError:
            sys.exit("grpc is required: pip install grpcio")
        try:
            servicer_cls, services_pb2_grpc = build_servicer_class(spec)
        except ImportError as e:
            sys.exit(
                f"lerobot is required for the async-inference wire types ({e}).\n"
                "Install it in this environment:\n"
                "  uv pip install --python .serve '.[serve]' --torch-backend cpu"
            )

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=args.workers),
            maximum_concurrent_rpcs=args.workers * 4,
            options=[
                ("grpc.max_receive_message_length", args.max_message_mb * 1024 * 1024),
                ("grpc.max_send_message_length", args.max_message_mb * 1024 * 1024),
                ("grpc.so_reuseport", 0),
            ])
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(
            servicer_cls(engine, ServerConfig(args)), server)
        # add_insecure_port returns 0 on failure; unchecked, the server logs
        # "listening" and blocks forever while nothing can connect.
        try:
            bound = server.add_insecure_port(f"{args.host}:{args.port}")
        except RuntimeError:
            bound = 0
        if bound == 0:
            sys.exit(f"failed to bind {args.host}:{args.port} (already in use?)")
        server.start()
        logger.info("vla.simd %s policy server listening on %s:%d",
                    spec.name, args.host, args.port)

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("shutting down")
        finally:
            # stop() first: freeing the model while a worker is inside
            # engine.predict is a use-after-free in the C library
            server.stop(grace=2.0).wait(timeout=10.0)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
