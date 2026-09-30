/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// vla.simd's own GGUF, written by tools/_gguf.py: text files as
// vla_simd.file.<path> strings, each .bin as a tensor <path>, or as parts
// <path>:0, <path>:1, ... when it mixes element types. The files come back
// byte for byte; no layout is decided here.

#include "io/gguf_models.h"
#include <cstdlib>
#include <filesystem>

namespace tcpu {
namespace io {

static bool relative_file(const std::string& name) {
    const std::filesystem::path p(name);
    // has_root_path, not is_absolute: on Windows "/x" and "C:x" are not absolute
    // but still escape the directory they are joined to.
    if (name.empty() || name.find('\0') != std::string::npos || p.has_root_path()) return false;
    for (const auto& part : p)
        if (part == ".." || part == "." || part.empty()) return false;
    return true;
}

bool adapt_vla_simd(const Gguf& g, Files& out, std::string& err) {
    const double format = g.num("vla_simd.format", 0);
    if (format != 1) {
        err = g.path() + ": vla.simd GGUF format " + std::to_string(format) +
              " is not supported (1 is); update vla.simd";
        return false;
    }
    const std::string prefix = "vla_simd.file.";
    for (const std::string& key : g.keys()) {
        if (key.compare(0, prefix.size(), prefix) != 0) continue;
        const std::string rel = key.substr(prefix.size());
        if (!relative_file(rel) || g.get(key)->type != GGUF_STRING) {
            err = g.path() + ": invalid embedded file " + rel;
            return false;
        }
        out[rel] = g.str(key);
    }

    // tensors in file order, so parts concatenate in the order they were written
    std::map<std::string, size_t> parts;
    for (const GgufTensor& t : g.tensors()) {
        if (!t.data) {
            err = g.path() + ": tensor " + t.name + " has an element type vla.simd does not write";
            return false;
        }
        std::string rel = t.name;
        const size_t colon = rel.rfind(':');
        if (colon != std::string::npos && colon + 1 < rel.size() &&
            rel.find_first_not_of("0123456789", colon + 1) == std::string::npos) {
            const std::string suffix = rel.substr(colon + 1);
            rel.resize(colon);
            const size_t part = parts[rel]++;
            if (suffix != std::to_string(part) || (part == 0 && out.count(rel))) {
                err = g.path() + ": invalid tensor part " + t.name;
                return false;
            }
        } else if (out.count(rel)) {
            err = g.path() + ": duplicate embedded file " + rel;
            return false;
        }
        if (!relative_file(rel)) {
            err = g.path() + ": invalid embedded file " + rel;
            return false;
        }
        out[rel].append((const char*)t.data, t.nbytes);
    }
    if (out.empty()) { err = g.path() + ": vla.simd GGUF holds no files"; return false; }
    return true;
}

} // namespace io
} // namespace tcpu
