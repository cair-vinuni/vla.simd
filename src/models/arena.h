/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "io/files.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace tcpu {

inline int metadata_int(double value) {
    if (!std::isfinite(value) || value < INT_MIN || value > INT_MAX || value != std::trunc(value))
        throw std::invalid_argument("invalid integer metadata");
    return static_cast<int>(value);
}

inline bool shape_fits(std::initializer_list<int> dims) {
    int count = 1;
    for (int dim : dims) {
        if (dim <= 0 || count > INT_MAX / dim) return false;
        count *= dim;
    }
    return true;
}

inline bool file_size_is(std::istream& file, size_t bytes) {
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0);
    return file && size >= 0 && static_cast<uintmax_t>(size) == bytes;
}

// Read a .bin into a float arena. Rejects a size that is not a whole number of
// floats: resize(bytes/4) then read(bytes) writes up to three bytes past the end.
inline bool read_arena(const std::string& path, std::vector<float>& data) {
    io::InFile bin(path, std::ios::binary);
    if (!bin) return false;
    bin.seekg(0, std::ios::end);
    const std::streamoff end = bin.tellg();
    bin.seekg(0);
    if (end < 0 || (size_t)end % sizeof(float) != 0) {
        std::fprintf(stderr, "%s: size %lld is not a whole number of floats\n",
                     path.c_str(), (long long)end);
        return false;
    }
    data.resize((size_t)end/sizeof(float));
    bin.read((char*)data.data(), (std::streamsize)(data.size()*sizeof(float)));
    return (bool)bin;
}

template<class T> struct ArenaCursor {
    const std::vector<T>& d;
    size_t off = 0;
    bool ok = true;
    const T* operator()(size_t n) {
        if (!ok || n > d.size() - off) { ok = false; return nullptr; }
        const T* p = d.data() + off;
        off += n;
        return p;
    }
    bool done() const { return ok && off == d.size(); }
};

inline bool read_floats(const std::string& path, std::vector<float>& out, size_t n) {
    io::InFile f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    if (n > SIZE_MAX / sizeof(float) || !file_size_is(f, n*sizeof(float))) return false;
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n*sizeof(float)));
    if (!f) return false;
    for (float value : out)
        if (!std::isfinite(value)) return false;
    return true;
}

} // namespace tcpu
