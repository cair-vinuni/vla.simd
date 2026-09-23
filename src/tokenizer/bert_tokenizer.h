/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include <unordered_map>
#include <vector>

// BERT WordPiece tokenizer (bert-base-uncased and friends). Loads a plain
// vocab.txt - one token per line, line number = id - as HF's BertTokenizer does.
//
// Two stages, same as the reference:
//   BasicTokenizer     clean control chars, normalize whitespace, pad CJK
//                      codepoints with spaces, lowercase + strip accents
//                      (do_lower_case), then split off punctuation.
//   WordpieceTokenizer greedy longest-match-first per word, continuations
//                      prefixed "##", [UNK] for a word that does not segment or
//                      is longer than max_chars.
//
// Lowercasing and accent stripping cover ASCII, Latin-1 Supplement and Latin
// Extended-A (through U+017F) plus the combining-mark block U+0300-U+036F, which
// is every codepoint an uncased English instruction can produce. Beyond that the
// text passes through unchanged and unmatched pieces become [UNK] - the same
// outcome an out-of-vocabulary word already gets, not silent corruption.

namespace tcpu {

struct BertTokenizer {
    std::unordered_map<std::string, int> vocab;   // token -> id
    bool lower_case = true;
    int max_chars = 100;                          // max_input_chars_per_word
    int cls_id = 101, sep_id = 102, pad_id = 0, unk_id = 100;

    // Reads <dir>/vocab.txt. The [CLS]/[SEP]/[PAD]/[UNK] ids are looked up in the
    // vocabulary, so a checkpoint that renumbers them still works.
    bool load(const std::string& dir);
    bool load_file(const std::string& vocab_path);

    // Word pieces of text, without [CLS]/[SEP].
    std::vector<int> pieces(const std::string& text) const;

    // [CLS] pieces [SEP], truncated to max_len ids in total (HF's
    // truncation=True, i.e. the [SEP] is kept), then right-padded with pad_id to
    // max_len when pad is true. n_real receives the count before padding.
    std::vector<int> encode(const std::string& text, int max_len, bool pad,
                            int* n_real = nullptr) const;
};

} // namespace tcpu
