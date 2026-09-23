/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include <unordered_map>
#include <vector>

// SentencePiece unigram tokenizer (T5 family). Loads vocab.txt (piece<TAB>score per
// line, line number = id) exported by tools/convert_t5_tokenizer.py. Viterbi
// segmentation over the normalized text. Normalization covers the practical ASCII
// subset of nmt_nfkc: whitespace collapse + dummy "meta symbol" prefix.

namespace tcpu {

struct T5Tokenizer {
    std::unordered_map<std::string, int> piece_id;
    std::vector<float> scores;
    int max_piece_len = 1;
    float min_score = 0.0f;
    int pad_id = 0, eos_id = 1, unk_id = 2;

    bool load(const std::string& dir);

    // Unigram-encode text; truncate pieces to max_len-1, append </s>, pad to max_len.
    std::vector<int> encode(const std::string& text, int max_len = 16) const;
};

} // namespace tcpu
