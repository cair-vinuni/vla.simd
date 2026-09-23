/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "t5_tokenizer.h"
#include "bert_tokenizer.h"
#include <fstream>
#include <limits>

namespace tcpu {

static const char* kSpace = "\xE2\x96\x81"; // sentencepiece meta symbol (U+2581)

static const uint32_t kCompat[][6] = {
    {0xA8, ' ', 0x308}, {0xAA, 'a'}, {0xAF, ' ', 0x304}, {0xB2, '2'}, {0xB3, '3'}, {0xB4, ' ', 0x301},
    {0xB5, 0x3BC}, {0xB8, ' ', 0x327}, {0xB9, '1'}, {0xBA, 'o'}, {0xBC, '1', 0x2044, '4'},
    {0xBD, '1', 0x2044, '2'}, {0xBE, '3', 0x2044, '4'}, {0x132, 'I', 'J'}, {0x133, 'i', 'j'},
    {0x13F, 'L', 0xB7}, {0x140, 'l', 0xB7}, {0x149, 0x2BC, 'n'}, {0x17F, 's'}, {0x340, 0x300},
    {0x341, 0x301}, {0x343, 0x313}, {0x344, 0x308, 0x301}, {0x2011, 0x2010}, {0x2017, ' ', 0x333},
    {0x2024, '.'}, {0x2025, '.', '.'}, {0x2026, '.', '.', '.'}, {0x2033, 0x2032, 0x2032},
    {0x2034, 0x2032, 0x2032, 0x2032}, {0x2036, 0x2035, 0x2035}, {0x2037, 0x2035, 0x2035, 0x2035},
    {0x203C, '!', '!'}, {0x203E, ' ', 0x305}, {0x2047, '?', '?'}, {0x2048, '?', '!'},
    {0x2049, '!', '?'}, {0x2057, 0x2032, 0x2032, 0x2032, 0x2032},
};

static bool is_deleted(uint32_t cp) {
    return (cp >= 1 && cp <= 8) || cp == 0xB || (cp >= 0xE && cp <= 0x1F) ||
           cp == 0x7F || cp == 0x8F || cp == 0x9F;
}

static bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\f' || cp == '\r' || cp == 0x85 ||
           cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200F) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x2581 || cp == 0x3000 ||
           cp == 0xFEFF || cp == 0xFFFD;
}

bool T5Tokenizer::load(const std::string& dir) {
    std::ifstream f(dir + "/vocab.txt");
    if (!f) return false;

    std::string line;
    int id = 0;
    min_score = 0.0;

    while (std::getline(f, line)) {
        size_t tab = line.rfind('\t');
        if (tab == std::string::npos) continue;

        std::string piece = line.substr(0, tab);
        double score      = std::stod(line.substr(tab+1));
        const int pid     = id++;
        scores.push_back(score);
        if (score == 0.0 && (piece.size() < 2 || piece[0] != '<' || piece.back() != '>')) continue;
        piece_id.emplace(piece, pid);

        if ((int)piece.size() > max_piece_len) max_piece_len = (int)piece.size();
        if (score < min_score) min_score = score;
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

void T5Tokenizer::encode_words(const std::string& text, std::vector<int>& ids) const {
    std::string s;
    bool space = true;
    auto put = [&](uint32_t cp) {
        if (is_space(cp)) { space = true; return; }
        if (space) { s += kSpace; space = false; }
        utf8_append(s, cp);
    };
    for (uint32_t cp : utf8_decode(text)) {
        if (is_deleted(cp)) continue;
        if (cp >= 0xFF01 && cp <= 0xFF5D) cp -= 0xFEE0;
        bool mapped = false;
        for (const auto& c : kCompat)
            if (c[0] == cp) { for (int k = 1; k < 6 && c[k]; k++) put(c[k]); mapped = true; }
        if (!mapped) put(cp);
    }

    // Viterbi over bytes: best[i] = max score of a segmentation of s[0:i)
    const int n            = (int)s.size();
    const double NEG       = -std::numeric_limits<double>::infinity();
    const double unk_score = min_score-10.0;
    std::vector<double> best (n+1, NEG);
    std::vector<int>    prev (n+1, -1);
    std::vector<int>    tok  (n+1, unk_id);
    best[0] = 0.0;

    for (int i=0; i<n; i++) {
        if (best[i] == NEG) continue;

        const int max_l = std::min(max_piece_len, n-i);
        for (int l=1; l<=max_l; l++) {
            auto it = piece_id.find(s.substr(i, l));
            if (it == piece_id.end()) continue;

            double sc = best[i]+scores[it->second];
            if (sc > best[i+l]) {
                best[i+l] = sc;
                prev[i+l] = i;
                tok[i+l]  = it->second;
            }
        }

        // unknown fallback: one UTF-8 char as <unk>
        int l = utf8_len((unsigned char)s[i]);
        if (best[i]+unk_score > best[i+l]) {
            best[i+l] = best[i]+unk_score;
            prev[i+l] = i;
            tok[i+l]  = unk_id;
        }
    }

    std::vector<int> rev;
    for (int i=n; i>0; i=prev[i])
        if (tok[i] != unk_id || rev.empty() || rev.back() != unk_id) rev.push_back(tok[i]);
    ids.insert(ids.end(), rev.rbegin(), rev.rend());
}

std::vector<int> T5Tokenizer::encode(const std::string& text, int max_len) const {
    if (max_len < 1) return {};   // the eos append below would overrun the caller

    std::vector<int> ids;
    size_t done = 0;
    for (size_t i = text.find('<'); i != std::string::npos; i = text.find('<', i+1)) {
        const size_t j = text.find('>', i);
        if (j == std::string::npos) break;
        if (j-i >= (size_t)max_piece_len) continue;
        auto it = piece_id.find(text.substr(i, j+1-i));
        if (it == piece_id.end() || scores[(size_t)it->second] != 0.0) continue;
        encode_words(text.substr(done, i-done), ids);
        ids.push_back(it->second);
        done = j+1;
    }
    encode_words(text.substr(done), ids);

    if ((int)ids.size() > max_len-1) ids.resize((size_t)(max_len-1));
    ids.push_back(eos_id);
    while ((int)ids.size() < max_len) ids.push_back(pad_id);
    return ids;
}

} // namespace tcpu
