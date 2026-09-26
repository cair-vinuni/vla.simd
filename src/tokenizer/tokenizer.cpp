/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tokenizer.h"
#include "io/files.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>

namespace tcpu {

static std::string cp_utf8(int cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else {
        s += (char)(0xC0 | (cp>>6));
        s += (char)(0x80 | (cp&0x3F));
    }
    return s;
}

static void build_byte2str(std::string out[256]) {
    std::vector<int> bs, cs;
    auto add = [&](int lo, int hi) {
        for (int b=lo; b<=hi; b++) {
            bs.push_back(b);
            cs.push_back(b);
        }
    };
    add(33, 126);
    add(161, 172);
    add(174, 255);

    int n = 0;
    for (int b=0; b<256; b++)
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256+n);
            n++;
        }

    for (size_t i=0; i<bs.size(); i++)
        out[bs[i]] = cp_utf8(cs[i]);
}

enum class Cls { Alpha, Digit, Space, Other };

static Cls cls_at(const std::string& s, size_t i, size_t e, size_t* len) {
    const unsigned char c = (unsigned char)s[i];
    *len = 1;
    if (c < 0x80) {
        if ((unsigned char)((c|32) - 'a') < 26) return Cls::Alpha;
        if ((unsigned char)(c - '0') < 10) return Cls::Digit;
        if (c == ' ' || (unsigned char)(c - '\t') < 5) return Cls::Space;
        return Cls::Other;
    }
    const size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
    if (!n || i + n > e) return Cls::Other;
    uint32_t cp = c & (0x7F >> n);
    for (size_t k = 1; k < n; k++) {
        const unsigned char d = (unsigned char)s[i+k];
        if ((d & 0xC0) != 0x80) return Cls::Other;
        cp = (cp << 6) | (d & 0x3F);
    }
    *len = n;
    static const uint32_t spaces[] = {0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000};
    for (uint32_t w : spaces) if (cp == w) return Cls::Space;
    if (cp >= 0x2000 && cp <= 0x200A) return Cls::Space;
    static const uint32_t numbers[][2] = {
        {0xB2, 0xB3}, {0xB9, 0xB9}, {0xBC, 0xBE}, {0x660, 0x669}, {0x6F0, 0x6F9}, {0x966, 0x96F},
        {0x2070, 0x2070}, {0x2074, 0x2079}, {0x2080, 0x2089}, {0x2150, 0x2182}, {0x2185, 0x2189},
        {0x2460, 0x249B}, {0x24EA, 0x24FF}, {0x2776, 0x2793}, {0x3007, 0x3007}, {0x3021, 0x3029},
        {0xFF10, 0xFF19},
    };
    for (auto& r : numbers) if (cp >= r[0] && cp <= r[1]) return Cls::Digit;
    static const uint32_t letters[][2] = {
        {0xAA, 0xAA}, {0xB5, 0xB5}, {0xBA, 0xBA}, {0xC0, 0xD6}, {0xD8, 0xF6}, {0xF8, 0x2C1},
        {0x370, 0x373}, {0x376, 0x377}, {0x37B, 0x37D}, {0x386, 0x386}, {0x388, 0x3FF},
        {0x400, 0x481}, {0x48A, 0x52F}, {0x531, 0x556}, {0x561, 0x587}, {0x5D0, 0x5EA},
        {0x620, 0x64A}, {0x1E00, 0x1FBC}, {0x3041, 0x3096}, {0x30A1, 0x30FA}, {0x3400, 0x4DBF},
        {0x4E00, 0x9FFF}, {0xAC00, 0xD7A3}, {0xF900, 0xFAFF}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A},
    };
    for (auto& r : letters) if (cp >= r[0] && cp <= r[1]) return Cls::Alpha;
    return Cls::Other;
}

static size_t run(const std::string& s, size_t i, size_t e, Cls want) {
    size_t n;
    while (i < e && cls_at(s, i, e, &n) == want) i += n;
    return i;
}

static size_t gpt2_piece(const std::string& s, size_t i, size_t e) {
    if (s[i] == '\'' && i+1 < e) {
        const char a = s[i+1], b = i+2 < e ? s[i+2] : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    size_t n;
    const Cls c = cls_at(s, i, e, &n);
    const size_t j = i + (s[i] == ' ');
    if (j < e) {
        const Cls cj = j == i ? c : cls_at(s, j, e, &n);
        if (cj == Cls::Alpha || cj == Cls::Digit || cj == Cls::Other) return run(s, j, e, cj) - i;
    }
    const size_t k = run(s, i, e, Cls::Space);
    if (k == e) return k - i;
    size_t last = i, m;
    for (size_t p = i; p < k; p += m) { cls_at(s, p, e, &m); last = p; }
    return last > i ? last - i : k - i;
}

bool Tokenizer::load(const std::string& dir) {
    build_byte2str(byte2str);

    io::InFile vf(dir + "/vocab.txt");
    if (!vf) {
        std::fprintf(stderr, "Tokenizer: cannot open %s/vocab.txt\n", dir.c_str());
        return false;
    }
    std::string line;
    while (std::getline(vf, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        vocab[line.substr(tab+1)] = std::stoi(line.substr(0, tab));
    }

    io::InFile mf(dir + "/merges.txt");
    if (!mf) {
        std::fprintf(stderr, "Tokenizer: cannot open %s/merges.txt\n", dir.c_str());
        return false;
    }
    int rank = 0;
    while (std::getline(mf, line)) {
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        ranks[{line.substr(0, sp), line.substr(sp+1)}] = rank++;
    }

    return !vocab.empty() && !ranks.empty();
}

std::vector<int> Tokenizer::bpe_encode(const std::string& piece_bytes) const {
    std::vector<std::string> word;
    for (unsigned char b : piece_bytes)
        word.push_back(byte2str[b]);

    while (word.size() > 1) {
        int best = std::numeric_limits<int>::max();
        int bi   = -1;
        for (size_t i=0; i+1<word.size(); i++) {
            auto it = ranks.find({word[i], word[i+1]});
            if (it != ranks.end() && it->second < best) {
                best = it->second;
                bi   = (int)i;
            }
        }
        if (bi < 0) break;

        word[bi] = word[bi]+word[bi+1];
        word.erase(word.begin()+bi+1);
    }

    std::vector<int> ids;
    for (auto& s : word) {
        auto it = vocab.find(s);
        if (it != vocab.end()) ids.push_back(it->second);
    }
    return ids;
}

void Tokenizer::encode_text(const std::string& text, std::vector<int>& ids) const {
    auto emit = [&](size_t i, size_t n) {
        auto sub = bpe_encode(text.substr(i, n));
        ids.insert(ids.end(), sub.begin(), sub.end());
    };
    const size_t N = text.size();
    for (size_t b = 0; b < N;) {
        size_t n;
        if (cls_at(text, b, N, &n) == Cls::Digit) { emit(b, n); b += n; continue; }
        size_t e = b;
        while (e < N && cls_at(text, e, N, &n) != Cls::Digit) e += n;
        for (size_t i = b; i < e;) {
            const size_t len = gpt2_piece(text, i, e);
            emit(i, len);
            i += len;
        }
        b = e;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    encode_text(text, ids);
    return ids;
}

} // namespace tcpu
