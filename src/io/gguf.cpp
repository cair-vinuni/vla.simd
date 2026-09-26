/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/gguf.h"
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tcpu {
namespace io {

int64_t GgufTensor::numel() const {
    int64_t n = 1;
    for (int64_t d : ne) n *= d;
    return n;
}

std::vector<int64_t> GgufTensor::shape() const {
    return std::vector<int64_t>(ne.rbegin(), ne.rend());
}

namespace {

// Every read is bounds-checked: the file is untrusted input, and a count field
// is the easiest way to walk a parser off the end of a mapping.
struct Cursor {
    const uint8_t* p;
    size_t n, off = 0;
    bool ok = true;

    bool need(size_t k) {
        if (!ok || k > n - off) ok = false;
        return ok;
    }
    template<class T> T get() {
        T v{};
        if (need(sizeof(T))) { std::memcpy(&v, p + off, sizeof(T)); off += sizeof(T); }
        return v;
    }
    std::string string() {
        const uint64_t len = get<uint64_t>();
        if (!need(len)) return {};
        std::string s((const char*)p + off, (size_t)len);
        off += len;
        return s;
    }
};

bool scalar(Cursor& c, uint32_t type, double& out) {
    switch (type) {
        case GGUF_U8:   out = c.get<uint8_t>();  return c.ok;
        case GGUF_I8:   out = c.get<int8_t>();   return c.ok;
        case GGUF_U16:  out = c.get<uint16_t>(); return c.ok;
        case GGUF_I16:  out = c.get<int16_t>();  return c.ok;
        case GGUF_U32:  out = c.get<uint32_t>(); return c.ok;
        case GGUF_I32:  out = c.get<int32_t>();  return c.ok;
        case GGUF_F32:  out = c.get<float>();    return c.ok;
        case GGUF_BOOL: out = c.get<uint8_t>() != 0; return c.ok;
        case GGUF_U64:  out = (double)c.get<uint64_t>(); return c.ok;
        case GGUF_I64:  out = (double)c.get<int64_t>();  return c.ok;
        case GGUF_F64:  out = c.get<double>();   return c.ok;
        default: return false;
    }
}

bool value(Cursor& c, uint32_t type, GgufValue& v) {
    v.type = type;
    if (type == GGUF_STRING) { v.str = c.string(); return c.ok; }
    if (type != GGUF_ARRAY) return scalar(c, type, v.num);

    v.elem = c.get<uint32_t>();
    const uint64_t n = c.get<uint64_t>();
    if (!c.ok || n > c.n) return false;          // each element takes at least a byte
    if (v.elem == GGUF_U8 || v.elem == GGUF_I8) {
        if (!c.need(n)) return false;
        v.bytes.assign(c.p + c.off, c.p + c.off + n);
        c.off += n;
        v.nums.assign(v.bytes.begin(), v.bytes.end());
        if (v.elem == GGUF_I8)
            for (size_t i = 0; i < n; i++) v.nums[i] = (int8_t)v.bytes[i];
        return true;
    }
    if (v.elem == GGUF_STRING) {
        v.strs.reserve((size_t)n);
        for (uint64_t i = 0; i < n && c.ok; i++) v.strs.push_back(c.string());
        return c.ok;
    }
    if (v.elem == GGUF_ARRAY) return false;      // nested arrays: never written by vla.cpp
    v.nums.resize((size_t)n);
    for (uint64_t i = 0; i < n; i++)
        if (!scalar(c, v.elem, v.nums[(size_t)i])) return false;
    return true;
}

size_t type_bytes(uint32_t type) {
    switch (type) {
        case GGML_F32:  return 4;
        case GGML_F16:  return 2;
        case GGML_BF16: return 2;
        case GGML_I8:   return 1;
        case GGML_I32:  return 4;
        default:        return 0;
    }
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {                                   // subnormal: renormalize
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            bits = sign | (exp << 23) | ((man & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) bits = sign | 0x7F800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

} // namespace

Gguf::~Gguf() {
    if (map) munmap(map, map_len);
}

bool Gguf::open(const std::string& path) {
    file = path;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "cannot open " + path; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 24) {
        ::close(fd);
        err = path + ": not a GGUF file (too short)";
        return false;
    }
    map_len = (size_t)st.st_size;
    map = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) { map = nullptr; err = "cannot map " + path; return false; }

    Cursor c{(const uint8_t*)map, map_len};
    if (std::memcmp(c.p, "GGUF", 4) != 0) { err = path + ": bad magic, not a GGUF file"; return false; }
    c.off = 4;
    const uint32_t version = c.get<uint32_t>();
    if (version < 2 || version > 3) {
        err = path + ": GGUF version " + std::to_string(version) + " is not supported (2 or 3)";
        return false;
    }
    const uint64_t n_tensors = c.get<uint64_t>();
    const uint64_t n_kv = c.get<uint64_t>();
    if (n_tensors > map_len || n_kv > map_len) { err = path + ": corrupt header counts"; return false; }

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = c.string();
        const uint32_t type = c.get<uint32_t>();
        GgufValue v;
        if (!c.ok || !value(c, type, v)) {
            err = path + ": corrupt metadata at key '" + key + "'";
            return false;
        }
        kv[key] = std::move(v);
    }

    struct Info { uint64_t offset; };
    std::vector<Info> offsets;
    table.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GgufTensor t;
        t.name = c.string();
        const uint32_t nd = c.get<uint32_t>();
        if (!c.ok || nd == 0 || nd > 4) { err = path + ": corrupt tensor info"; return false; }
        t.ne.resize(nd);
        for (uint32_t d = 0; d < nd; d++) {
            const uint64_t v = c.get<uint64_t>();
            if (v == 0 || v > (uint64_t)1 << 40) { err = path + ": bad shape for " + t.name; return false; }
            t.ne[d] = (int64_t)v;
        }
        t.type = c.get<uint32_t>();
        offsets.push_back({c.get<uint64_t>()});
        if (!c.ok) { err = path + ": corrupt tensor info"; return false; }
        if (index.count(t.name)) { err = path + ": duplicate tensor " + t.name; return false; }
        index[t.name] = table.size();
        table.push_back(std::move(t));
    }

    const size_t align = (size_t)num("general.alignment", 32);
    if (align == 0 || (align & (align - 1))) { err = path + ": bad general.alignment"; return false; }
    const size_t base = (c.off + align - 1) / align * align;
    for (size_t i = 0; i < table.size(); i++) {
        GgufTensor& t = table[i];
        const size_t eb = type_bytes(t.type);
        // Quantized types have no fixed element size; they are rejected when
        // widened, but their extent is unknown, so they get no data pointer.
        if (eb == 0) continue;
        t.nbytes = (size_t)t.numel() * eb;
        const uint64_t off = offsets[i].offset;
        if (off > map_len || base > map_len - off || t.nbytes > map_len - base - off) {
            err = path + ": tensor " + t.name + " runs past the end of the file";
            return false;
        }
        t.data = (const uint8_t*)map + base + off;
    }
    return true;
}

std::vector<std::string> Gguf::keys() const {
    std::vector<std::string> out;
    for (const auto& kv_ : kv) out.push_back(kv_.first);
    return out;
}

const GgufValue* Gguf::get(const std::string& key) const {
    auto it = kv.find(key);
    return it == kv.end() ? nullptr : &it->second;
}

std::string Gguf::str(const std::string& key, const std::string& dflt) const {
    const GgufValue* v = get(key);
    return v && v->type == GGUF_STRING ? v->str : dflt;
}

double Gguf::num(const std::string& key, double dflt) const {
    const GgufValue* v = get(key);
    return v && v->type != GGUF_STRING && v->type != GGUF_ARRAY ? v->num : dflt;
}

const GgufTensor* Gguf::tensor(const std::string& name) const {
    auto it = index.find(name);
    return it == index.end() ? nullptr : &table[it->second];
}

bool tensor_f32(const GgufTensor& t, std::vector<float>& out, std::string& err) {
    const size_t n = (size_t)t.numel();
    if (!t.data || t.type == GGML_I8 || t.type == GGML_I32) {
        err = t.name + ": element type " + std::to_string(t.type) +
              " is not supported (F32, F16 and BF16 are)";
        return false;
    }
    out.resize(n);
    if (t.type == GGML_F32) {
        std::memcpy(out.data(), t.data, n * 4);
    } else if (t.type == GGML_BF16) {
        for (size_t i = 0; i < n; i++) {
            uint16_t h;
            std::memcpy(&h, t.data + 2*i, 2);
            const uint32_t bits = (uint32_t)h << 16;
            std::memcpy(&out[i], &bits, 4);
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            uint16_t h;
            std::memcpy(&h, t.data + 2*i, 2);
            out[i] = f16_to_f32(h);
        }
    }
    return true;
}

uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t lsb = (u >> 16) & 1;
    return (uint16_t)((u + 0x7FFF + lsb) >> 16);
}

} // namespace io
} // namespace tcpu
