/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Llama-3 byte-level BPE tokenizer (the BitVLA tokenizer). Modeled on TinyChatEngine
// OPTTokenizer (byte-level BPE) with Llama-3 specifics: the GPT-4 pretokenizer regex
// (ASCII implementation), ignore_merges (tiktoken direct-vocab shortcut), and the
// 128k special tokens. Loads vocab/merges/specials exported by tools/convert_tokenizer.py.

namespace tcpu {

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;             // byte-level token -> id
    std::map<std::pair<std::string, std::string>, int> ranks; // merge pair -> rank
    std::unordered_map<std::string, int> specials;          // content -> id
    std::string byte2str[256];
    int bos_id = 128000;

    bool load(const std::string& dir);

    // Encode text (no special-token splitting) to ids; prepends bos if add_bos.
    std::vector<int> encode(const std::string& text, bool add_bos = false) const;

    // Encode text, splitting on special-token content (emitting their ids); bos optional.
    std::vector<int> encode_with_specials(const std::string& text, bool add_bos = false) const;

private:
    std::vector<int> bpe_encode(const std::string& piece_bytes) const;
    void encode_text(const std::string& text, std::vector<int>& ids) const;
};

} // namespace tcpu
