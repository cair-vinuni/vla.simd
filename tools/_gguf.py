"""
Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
Licensed under the Apache License, Version 2.0.
SPDX-License-Identifier: Apache-2.0

The vla.simd GGUF: one file per converted checkpoint (numpy only, no torch, no
gguf package).

A converter still lays its weights out exactly as the engine's loaders consume
them; it writes those files into a private staging directory, and pack() turns
the directory into one GGUF:

  general.architecture   "vla-simd"
  vla_simd.model         act | impact | diffusion | smolvla | turbovla | octo
  vla_simd.format        1
  vla_simd.source        the checkpoint it was converted from
  vla_simd.file.<path>   every text file (.meta, config.txt, tokenizer tables)
  tensor <path>          every .bin, 1-D, typed (F32 unless the layout says
                         otherwise); a file mixing types is split into parts
                         <path>:0, <path>:1, ... concatenated in order

The engine (src/io/gguf_models.cpp) rebuilds the same files byte for byte, so a
GGUF loads exactly like the directory it replaces.

    with gguf_output(args.out, "act", source=args.ckpt) as out:
        dump_x(..., out.dir)                     # unchanged file-writing code
        out.layout["emb.bin"] = [("BF16", None)]  # when a .bin is not fp32
"""

import contextlib
import math
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import PurePosixPath


FORMAT = 1
ALIGN = 32
TYPES = {"F32": (0, 4), "BF16": (30, 2), "I8": (24, 1), "I32": (26, 4)}
_KV_STRING, _KV_U32 = 8, 4


def out_path(out):
    """--out as a file: a .gguf path as given, else <dir>/<dir name>.gguf."""
    out = os.path.expanduser(out)
    if out.endswith(".gguf"):
        path = out
    else:
        out = out.rstrip("/")
        if os.path.isdir(out) and any(f.endswith(".meta") for f in os.listdir(out)):
            sys.exit(f"{out} holds a converted directory (.meta files); a GGUF beside them "
                     "would not load. Pass another --out, or a .gguf path.")
        path = os.path.join(out, os.path.basename(out) + ".gguf")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


class Output:
    def __init__(self, directory):
        self.dir = directory
        self.layout = {}          # relative .bin path -> [(type, nbytes or None for the rest)]


@contextlib.contextmanager
def gguf_output(out, model, source=""):
    """Stage the converter's files, then pack them into out_path(out)."""
    path = out_path(out)
    staging = tempfile.mkdtemp(prefix=".vla_simd_", dir=os.path.dirname(os.path.abspath(path)))
    try:
        o = Output(staging)
        yield o
        n = pack(staging, path, model, source, o.layout)
        print(f"wrote {path} ({n / 1e6:.1f} MB)", flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _files(root):
    for d, dirs, names in os.walk(root):
        dirs.sort()
        for n in sorted(names):
            p = os.path.join(d, n)
            yield os.path.relpath(p, root).replace(os.sep, "/"), p


def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _publish(tmp, path):
    """os.replace, retried on Windows: there, replacing a file that another
    writer is replacing at the same moment is refused (PermissionError) rather
    than serialized."""
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 49:
                raise
            time.sleep(0.02)


def pack(root, path, model, source, layout):
    kv = [("general.architecture", _KV_STRING, "vla-simd"),
          ("general.name", _KV_STRING, os.path.splitext(os.path.basename(path))[0]),
          ("vla_simd.model", _KV_STRING, model),
          ("vla_simd.format", _KV_U32, FORMAT),
          ("vla_simd.source", _KV_STRING, str(source or ""))]
    tensors = []
    for rel, p in _files(root):
        length = os.path.getsize(p)
        if not rel.endswith(".bin") or not length:
            with open(p, "rb") as f:
                data = f.read()
            try:
                kv.append((f"vla_simd.file.{rel}", _KV_STRING, data.decode("utf-8")))
            except UnicodeDecodeError:
                sys.exit(f"{rel}: not UTF-8 text; name binary files *.bin")
            continue
        parts = layout.get(rel, [("F32", None)])
        at = 0
        for i, (ty, n) in enumerate(parts):
            n = length - at if n is None else n
            tid, size = TYPES[ty]
            if not isinstance(n, int) or n <= 0 or n % size or n > length - at:
                raise ValueError(f"{rel}: invalid {ty} part {i} of {n} bytes")
            name = rel if len(parts) == 1 else f"{rel}:{i}"
            tensors.append((name, tid, n // size, p, at, n))
            at += n
        if at != length:
            raise ValueError(f"{rel}: layout covers {at} of {length} bytes")

    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for key, ty, val in kv:
        head += _str(key) + struct.pack("<I", ty)
        head += _str(val) if ty == _KV_STRING else struct.pack("<I", val)
    offsets, off = [], 0
    for name, tid, n, _, _, length in tensors:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        offsets.append(off)
        head += _str(name) + struct.pack("<IQIQ", 1, n, tid, off)
        off += length
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(path)), delete=False) as f:
            tmp = f.name
            f.write(head)
            f.write(b"\0" * (-len(head) % ALIGN))
            base = f.tell()
            for (_, _, _, src, start, length), o in zip(tensors, offsets):
                f.write(b"\0" * (base + o - f.tell()))
                with open(src, "rb") as inp:
                    inp.seek(start)
                    while length:
                        chunk = inp.read(min(length, 1 << 20))
                        if not chunk:
                            raise ValueError(f"{src}: truncated during packing")
                        f.write(chunk)
                        length -= len(chunk)
        _publish(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return os.path.getsize(path)


def read_files(path):
    """A vla.simd GGUF back as {relative path: bytes}, e.g. to reuse a base
    checkpoint's frozen tower."""
    with open(path, "rb") as f:
        buf = f.read()
    o = 0

    def rd(fmt):
        nonlocal o
        if struct.calcsize(fmt) > len(buf) - o:
            raise ValueError(f"{path}: truncated GGUF")
        v = struct.unpack_from(fmt, buf, o)
        o += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def rs():
        nonlocal o
        n = rd("<Q")
        if n > len(buf) - o:
            raise ValueError(f"{path}: truncated string")
        s = buf[o:o + n]
        o += n
        return s.decode("utf-8")

    if buf[:4] != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    o = 4
    version, n_t, n_kv = rd("<IQQ")
    if version not in (2, 3) or n_t > len(buf) // 24 or n_kv > len(buf) // 12:
        raise ValueError(f"{path}: invalid GGUF header")
    files, arch = {}, None
    alignment, format_version, keys = ALIGN, None, set()

    def file_name(name):
        if not name or "\0" in name or PurePosixPath(name).is_absolute() or any(
                p in ("", ".", "..") for p in name.split("/")):
            raise ValueError(f"{path}: invalid embedded path {name!r}")
        return name
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
              10: "<Q", 11: "<q", 12: "<d"}
    for _ in range(n_kv):
        key, ty = rs(), rd("<I")
        if key in keys:
            raise ValueError(f"{path}: duplicate metadata {key}")
        keys.add(key)
        if ty == _KV_STRING:
            val = rs()
            if key == "general.architecture":
                arch = val
            elif key.startswith("vla_simd.file."):
                files[file_name(key[len("vla_simd.file."):])] = val.encode("utf-8")
        elif ty == 9:
            raise ValueError(f"{path}: arrays are not part of the vla.simd format")
        else:
            if ty not in scalar:
                raise ValueError(f"{path}: invalid metadata type {ty}")
            val = rd(scalar[ty])
            if key == "general.alignment":
                if ty != _KV_U32 or val == 0 or val & (val - 1):
                    raise ValueError(f"{path}: invalid alignment")
                alignment = val
            elif key == "vla_simd.format":
                format_version = val
    if arch != "vla-simd":
        raise ValueError(f"{path} is a '{arch}' GGUF, not a vla.simd one")
    if format_version != FORMAT:
        raise ValueError(f"{path}: unsupported vla.simd format {format_version}")
    infos = []
    names = set()
    for _ in range(n_t):
        name, nd = rs(), rd("<I")
        if name in names or not 1 <= nd <= 4:
            raise ValueError(f"{path}: invalid tensor info {name}")
        names.add(name)
        ne = [rd("<Q") for _ in range(nd)]
        ty, off = rd("<IQ")
        size = next((s for t, s in TYPES.values() if t == ty), None)
        if size is None or any(d == 0 for d in ne) or off % alignment:
            raise ValueError(f"{path}: invalid tensor {name}")
        infos.append((name, math.prod(ne) * size, off))
    base = (o + alignment - 1) // alignment * alignment
    if base > len(buf):
        raise ValueError(f"{path}: truncated alignment padding")
    parts = {}
    for name, nbytes, off in infos:
        if base + off + nbytes > len(buf):
            raise ValueError(f"{path}: truncated tensor {name}")
        rel = name.rsplit(":", 1)[0] if ":" in name and name.rsplit(":", 1)[1].isdigit() else name
        file_name(rel)
        if rel != name:
            part = parts.get(rel, 0)
            if name != f"{rel}:{part}" or (part == 0 and rel in files):
                raise ValueError(f"{path}: invalid tensor part {name}")
            parts[rel] = part + 1
        elif rel in files:
            raise ValueError(f"{path}: duplicate embedded file {rel}")
        files[rel] = files.get(rel, b"") + buf[base + off:base + off + nbytes]
    return files
