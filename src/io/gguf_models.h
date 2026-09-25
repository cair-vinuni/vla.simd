/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>
#include "io/gguf.h"

// vla.cpp GGUF -> the converted-directory files the loaders read. One adapter
// per architecture, each a C++ port of the Python converter for that model
// (tools/convert_*.py), fed from GGUF tensors instead of the upstream checkpoint.

namespace tcpu {
namespace io {

using Files = std::map<std::string, std::string>;   // relative path -> bytes

// A file beside the GGUF that the GGUF does not carry.
struct Sidecar {
    std::string dir;
    bool read(const std::string& rel, std::string& out) const;
    // key/value lines of <dir>/config.txt, split on the first space
    std::map<std::string, std::string> config() const;
};

// The sidecar config.txt as written, followed by each of `add` it does not set.
std::string config_with(const Sidecar& side,
                        std::initializer_list<std::pair<const char*, std::string>> add);

bool adapt_gguf(const Gguf& g, const Sidecar& side, Files& out, std::string& err);

bool adapt_smolvla(const Gguf& g, const Sidecar& side, Files& out, std::string& err);
bool adapt_turbovla(const Gguf& g, const Sidecar& side, Files& out, std::string& err);
bool adapt_octo(const Gguf& g, const Sidecar& side, Files& out, std::string& err);
// vla.simd's own GGUF (tools/_gguf.py): the converted files, stored verbatim
bool adapt_vla_simd(const Gguf& g, Files& out, std::string& err);

// ---------------------------------------------------------------------------
// shared by the adapters
// ---------------------------------------------------------------------------

// Tensor access with the shape checked against the PyTorch-order shape the
// converter would have seen. The first failure sticks in `err`; later calls are
// no-ops, so an adapter can read a whole arena and test once at the end.
struct TensorReader {
    const Gguf& g;
    std::string& err;

    bool ok() const { return err.empty(); }
    const GgufTensor* find(const std::string& name);
    std::vector<int64_t> shape(const std::string& name);
    // fp32 values; `want` (PyTorch order) is checked when given
    std::vector<float> f32(const std::string& name, std::initializer_list<int64_t> want = {});
    uint32_t u32(const std::string& key);
    double num(const std::string& key);
};

// Byte blob, appended in the order the engine's ArenaCursor consumes it.
struct Blob {
    std::string bytes;
    void f32(const std::vector<float>& v) { raw(v.data(), v.size() * 4); }
    void f32(const float* p, size_t n) { raw(p, n * 4); }
    void bf16(const std::vector<float>& v);          // fp32 -> bf16 (RNE)
    void raw(const void* p, size_t n) { bytes.append((const char*)p, n); }
};

// Python's repr() of a float (of a float32 when `single`).
std::string pyrepr(double v, bool single = false);

// "key value\n" lines. Floats go out with enough digits to read back exactly.
struct Meta {
    std::string text;
    Meta& i(const char* key, long long v);
    Meta& f(const char* key, double v);
    Meta& s(const char* key, const std::string& v);
};

} // namespace io
} // namespace tcpu
