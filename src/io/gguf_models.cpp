/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/gguf_models.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tcpu {
namespace io {

namespace {

std::string shape_str(const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); i++) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

} // namespace

// Python's repr(): the shortest decimal that reads back to the same value,
// positional between 1e-4 and 1e16, so a .meta line says "1e-06" and "100.0"
// the way the Python converters wrote them.
std::string pyrepr(double v, bool single) {
    char buf[64];
    int p = 1;
    for (; p < 17; p++) {
        std::snprintf(buf, sizeof buf, "%.*g", p, v);
        const double back = std::strtod(buf, nullptr);
        if (single ? (float)back == (float)v : back == v) break;
    }
    const double a = std::fabs(v);
    if (v == 0 || (a >= 1e-4 && a < 1e16)) {
        const int e = v == 0 ? 0 : (int)std::floor(std::log10(a));
        std::snprintf(buf, sizeof buf, "%.*f", std::max(0, p - 1 - e), v);
        std::string s = buf;
        if (s.find('.') == std::string::npos) s += ".0";
        return s;
    }
    std::snprintf(buf, sizeof buf, "%.*g", p, v);
    return buf;
}


bool Sidecar::read(const std::string& rel, std::string& out) const {
    std::ifstream f(dir + "/" + rel, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::map<std::string, std::string> Sidecar::config() const {
    std::map<std::string, std::string> cfg;
    std::string text;
    if (!read("config.txt", text)) return cfg;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        cfg[line.substr(0, sp)] = line.substr(sp + 1);
    }
    return cfg;
}

std::string config_with(const Sidecar& side,
                        std::initializer_list<std::pair<const char*, std::string>> add) {
    std::string text;
    side.read("config.txt", text);
    if (!text.empty() && text.back() != '\n') text += '\n';
    const std::map<std::string, std::string> have = side.config();
    for (const auto& [k, v] : add)
        if (!have.count(k)) text += std::string(k) + " " + v + "\n";
    return text;
}

const GgufTensor* TensorReader::find(const std::string& name) {
    if (!ok()) return nullptr;
    const GgufTensor* t = g.tensor(name);
    if (!t) err = g.path() + ": tensor " + name + " not found";
    return t;
}

std::vector<int64_t> TensorReader::shape(const std::string& name) {
    const GgufTensor* t = find(name);
    return t ? t->shape() : std::vector<int64_t>{};
}

std::vector<float> TensorReader::f32(const std::string& name, std::initializer_list<int64_t> want) {
    std::vector<float> out;
    const GgufTensor* t = find(name);
    if (!t) return out;
    if (want.size()) {
        // Leading 1s are how a squeezed tensor and its unsqueezed source differ
        // (view_emb [1,2,256] vs [2,256]); the element order is the same.
        std::vector<int64_t> have = t->shape(), need(want);
        while (have.size() > 1 && have.front() == 1) have.erase(have.begin());
        while (need.size() > 1 && need.front() == 1) need.erase(need.begin());
        if (have != need) {
            err = g.path() + ": tensor " + name + " is " + shape_str(t->shape()) +
                  ", expected " + shape_str(std::vector<int64_t>(want));
            return out;
        }
    }
    if (!tensor_f32(*t, out, err)) err = g.path() + ": " + err;
    return out;
}

double TensorReader::num(const std::string& key) {
    if (!ok()) return 0;
    const GgufValue* v = g.get(key);
    if (!v || v->type == GGUF_STRING || v->type == GGUF_ARRAY) {
        err = g.path() + ": metadata " + key + " missing";
        return 0;
    }
    // A float32 key holds 1e-6 as 9.99999997e-07; give back the double the
    // converter would have used, which is the short decimal it was written from.
    if (v->type == GGUF_F32) return std::strtod(pyrepr(v->num, true).c_str(), nullptr);
    return v->num;
}

uint32_t TensorReader::u32(const std::string& key) {
    const double v = num(key);
    if (ok() && (v < 0 || v != (double)(uint32_t)v)) err = g.path() + ": metadata " + key + " is not a count";
    return (uint32_t)v;
}

void Blob::bf16(const std::vector<float>& v) {
    const size_t at = bytes.size();
    bytes.resize(at + v.size() * 2);
    for (size_t i = 0; i < v.size(); i++) {
        const uint16_t h = f32_to_bf16(v[i]);
        std::memcpy(&bytes[at + 2*i], &h, 2);
    }
}

Meta& Meta::i(const char* key, long long v) {
    text += std::string(key) + " " + std::to_string(v) + "\n";
    return *this;
}

Meta& Meta::f(const char* key, double v) {
    text += std::string(key) + " " + pyrepr(v) + "\n";
    return *this;
}

Meta& Meta::s(const char* key, const std::string& v) {
    text += std::string(key) + " " + v + "\n";
    return *this;
}

bool adapt_gguf(const Gguf& g, const Sidecar& side, Files& out, std::string& err) {
    const std::string arch = g.str("general.architecture");
    if (arch == "smolvla")  return adapt_smolvla(g, side, out, err);
    if (arch == "turbovla") return adapt_turbovla(g, side, out, err);
    if (arch == "octo")     return adapt_octo(g, side, out, err);
    if (arch == "vla-simd") return adapt_vla_simd(g, out, err);
    err = g.path() + ": GGUF architecture '" + arch +
          "' is not supported (smolvla, turbovla and octo are)";
    return false;
}

} // namespace io
} // namespace tcpu
