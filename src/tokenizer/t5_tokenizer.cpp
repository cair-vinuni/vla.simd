/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "t5_tokenizer.h"
#include <fstream>
#include <limits>

namespace tcpu {

static const char* kSpace = "\xE2\x96\x81"; // sentencepiece meta symbol (U+2581)

bool T5Tokenizer::load(const std::string& dir) {
    std::ifstream f(dir + "/vocab.txt");
    if (!f) return false;

    std::string line;
    int id = 0;
    min_score = 0.0f;

    while (std::getline(f, line)) {
        size_t tab = line.rfind('\t');
        if (tab == std::string::npos) continue;

        std::string piece = line.substr(0, tab);
        float score       = std::stof(line.substr(tab+1));
        piece_id.emplace(piece, id);
        scores.push_back(score);

        if ((int)piece.size() > max_piece_len) max_piece_len = (int)piece.size();
        if (score < min_score) min_score = score;
        id++;
    }
    return id > 0;
}

static int utf8_len(unsigned char c) {
    if (c < 0x80      ) return 1;
    if ((c>>5) == 0x6 ) return 2;
    if ((c>>4) == 0xE ) return 3;
    if ((c>>3) == 0x1E) return 4;
    return 1;
}

std::vector<int> T5Tokenizer::encode(const std::string& text, int max_len) const {
    if (max_len < 1) return {};   // the eos append below would overrun the caller

    // normalize: collapse whitespace runs, trim, ' ' -> meta symbol, dummy prefix
    std::string s;
    s.reserve(text.size()*3+3);
    bool at_start      = true;
    bool pending_space = false;

    for (char ch : text) {
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!at_start) pending_space = true;
            continue;
        }
        if (pending_space) {
            s += kSpace;
            pending_space = false;
        }
        if (at_start) {
            s += kSpace;
            at_start = false;
        }
        s += ch;
    }

    // Viterbi over bytes: best[i] = max score of a segmentation of s[0:i)
    const int n           = (int)s.size();
    const float NEG       = -std::numeric_limits<float>::infinity();
    const float unk_score = min_score-10.0f;
    std::vector<float> best (n+1, NEG);
    std::vector<int>   prev (n+1, -1);
    std::vector<int>   tok  (n+1, unk_id);
    best[0] = 0.0f;

    for (int i=0; i<n; i++) {
        if (best[i] == NEG) continue;

        const int max_l = std::min(max_piece_len, n-i);
        for (int l=1; l<=max_l; l++) {
            auto it = piece_id.find(s.substr(i, l));
            if (it == piece_id.end()) continue;

            float sc = best[i]+scores[it->second];
            if (sc > best[i+l]) {
                best[i+l] = sc;
                prev[i+l] = i;
                tok[i+l]  = it->second;
            }
        }

        // unknown fallback: one UTF-8 char as <unk>. Clamped to the remaining
        // bytes: a truncated trailing sequence otherwise leaves best[n] at NEG,
        // and the backtrace then collapses the whole string to one <unk>.
        int l = utf8_len((unsigned char)s[i]);
        if (l > n-i) l = n-i;
        if (best[i]+unk_score > best[i+l]) {
            best[i+l] = best[i]+unk_score;
            prev[i+l] = i;
            tok[i+l]  = unk_id;
        }
    }

    std::vector<int> rev;
    for (int i=n; i>0; i=prev[i])
        rev.push_back(tok[i]);

    std::vector<int> ids;
    ids.reserve(max_len);
    for (int i=(int)rev.size()-1; i>=0 && (int)ids.size() < max_len-1; i--)
        ids.push_back(rev[i]);

    ids.push_back(eos_id);
    while ((int)ids.size() < max_len) ids.push_back(pad_id);
    return ids;
}

} // namespace tcpu
