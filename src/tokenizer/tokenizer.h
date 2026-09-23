/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tcpu {

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;             // byte-level token -> id
    std::map<std::pair<std::string, std::string>, int> ranks; // merge pair -> rank
    std::string byte2str[256];

    bool load(const std::string& dir);

    // Encode text (no special-token splitting) to ids.
    std::vector<int> encode(const std::string& text) const;

private:
    std::vector<int> bpe_encode(const std::string& piece_bytes) const;
    void encode_text(const std::string& text, std::vector<int>& ids) const;
};

} // namespace tcpu
