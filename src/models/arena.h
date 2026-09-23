/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace tcpu {

// Read a .bin into a float arena. Rejects a size that is not a whole number of
// floats: resize(bytes/4) then read(bytes) writes up to three bytes past the end.
inline bool read_arena(const std::string& path, std::vector<float>& data) {
    std::ifstream bin(path, std::ios::binary);
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

} // namespace tcpu
