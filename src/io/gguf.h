/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Read-only GGUF (v2/v3) reader: header, metadata and tensor table, with the
// tensor data mapped rather than copied. No ggml dependency; only the element
// types a vla.cpp checkpoint actually ships (F32, F16, BF16) can be widened.

namespace tcpu {
namespace io {

enum GgufType : uint32_t {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5,
    GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10,
    GGUF_I64 = 11, GGUF_F64 = 12,
};

enum GgmlType : uint32_t { GGML_F32 = 0, GGML_F16 = 1, GGML_BF16 = 30 };

struct GgufValue {
    uint32_t type = 0;
    uint32_t elem = 0;               // element type of an array
    double num = 0;                  // any scalar number or bool
    std::string str;
    std::vector<double> nums;        // numeric array
    std::vector<std::string> strs;   // string array
    std::vector<uint8_t> bytes;      // u8/i8 array, kept raw (e.g. a sentencepiece proto)
};

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<int64_t> ne;         // ggml order: ne[0] is the contiguous dimension
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    int64_t numel() const;
    // PyTorch-order shape (ne reversed), for error messages and shape checks
    std::vector<int64_t> shape() const;
};

class Gguf {
public:
    Gguf() = default;
    ~Gguf();
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;

    bool open(const std::string& path);
    const std::string& error() const { return err; }
    const std::string& path() const { return file; }

    bool has(const std::string& key) const { return kv.count(key) != 0; }
    const GgufValue* get(const std::string& key) const;
    std::string str(const std::string& key, const std::string& dflt = "") const;
    double num(const std::string& key, double dflt) const;

    const GgufTensor* tensor(const std::string& name) const;
    const std::vector<GgufTensor>& tensors() const { return table; }

private:
    std::string file, err;
    void* map = nullptr;
    size_t map_len = 0;
    std::map<std::string, GgufValue> kv;
    std::vector<GgufTensor> table;
    std::map<std::string, size_t> index;
};

// Widen a tensor to fp32. BF16 and F16 widen exactly.
bool tensor_f32(const GgufTensor& t, std::vector<float>& out, std::string& err);
// fp32 -> bf16 with round-to-nearest-even, bit for bit what the converters do.
uint16_t f32_to_bf16(float x);

} // namespace io
} // namespace tcpu
