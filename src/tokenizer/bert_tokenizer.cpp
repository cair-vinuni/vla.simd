/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bert_tokenizer.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace tcpu {

// ---------------------------------------------------------------- UTF-8 -----
std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i++];
        int extra = c < 0x80 ? 0 : c < 0xC2 ? -1 : c < 0xE0 ? 1 : c < 0xF0 ? 2 : c < 0xF5 ? 3 : -1;
        if (extra < 0) { cps.push_back(0xFFFD); continue; }
        uint32_t cp = c & (0x7Fu >> extra);
        unsigned char lo = c == 0xE0 ? 0xA0 : c == 0xF0 ? 0x90 : 0x80;
        unsigned char hi = c == 0xED ? 0x9F : c == 0xF4 ? 0x8F : 0xBF;
        while (extra > 0 && i < s.size() && (unsigned char)s[i] >= lo && (unsigned char)s[i] <= hi) {
            cp = (cp << 6) | ((unsigned char)s[i++] & 0x3Fu);
            extra--; lo = 0x80; hi = 0xBF;
        }
        cps.push_back(extra ? 0xFFFD : cp);
    }
    return cps;
}

void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

namespace {

// ------------------------------------------------- character predicates -----
template <size_t N> bool in_ranges(uint32_t cp, const uint32_t (&r)[N][2]) {
    for (const auto& x : r) if (cp >= x[0] && cp <= x[1]) return true;
    return false;
}

bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_control(uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r' || (cp >= 0x20 && cp < 0x7F)) return false;
    static const uint32_t r[][2] = {
        {0x00, 0x1F}, {0x7F, 0x9F}, {0xAD, 0xAD}, {0x600, 0x605}, {0x61C, 0x61C}, {0x6DD, 0x6DD},
        {0x70F, 0x70F}, {0x180E, 0x180E}, {0x200B, 0x200F}, {0x202A, 0x202E}, {0x2060, 0x2064},
        {0x2066, 0x206F}, {0xE000, 0xF8FF}, {0xFEFF, 0xFEFF}, {0xFFF9, 0xFFFB}, {0x110BD, 0x110BD},
        {0x1BCA0, 0x1BCA3}, {0x1D173, 0x1D17A}, {0xE0001, 0xE0001}, {0xE0020, 0xE007F},
        {0xF0000, 0xFFFFD}, {0x100000, 0x10FFFD},
    };
    return in_ranges(cp, r);
}

bool is_punctuation(uint32_t cp) {
    // The reference treats every ASCII non-alphanumeric as punctuation, plus
    // anything in Unicode category P. The P set below is the Latin-1, general
    // punctuation, CJK and fullwidth blocks - what real instruction text uses.
    if (cp < 0x80) return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
                          (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
    static const uint32_t r[][2] = {
        {0xA1, 0xA1}, {0xA7, 0xA7}, {0xAB, 0xAB}, {0xB6, 0xB7}, {0xBB, 0xBB}, {0xBF, 0xBF},
        {0x2010, 0x2027}, {0x2030, 0x2043}, {0x2045, 0x2051}, {0x2053, 0x205E}, {0x207D, 0x207E},
        {0x208D, 0x208E}, {0x2768, 0x2775}, {0x2E00, 0x2E2E}, {0x2E30, 0x2E42}, {0x3001, 0x3003},
        {0x3008, 0x3011}, {0x3014, 0x301F}, {0x3030, 0x3030}, {0x303D, 0x303D}, {0x30A0, 0x30A0},
        {0x30FB, 0x30FB}, {0xFE10, 0xFE19}, {0xFE30, 0xFE52}, {0xFE54, 0xFE61}, {0xFE63, 0xFE63},
        {0xFE68, 0xFE68}, {0xFE6A, 0xFE6B}, {0xFF01, 0xFF03}, {0xFF05, 0xFF0A}, {0xFF0C, 0xFF0F},
        {0xFF1A, 0xFF1B}, {0xFF1F, 0xFF20}, {0xFF3B, 0xFF3D}, {0xFF3F, 0xFF3F}, {0xFF5B, 0xFF5B},
        {0xFF5D, 0xFF5D}, {0xFF5F, 0xFF65},
    };
    return in_ranges(cp, r);
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00  && cp <= 0x9FFF ) || (cp >= 0x3400  && cp <= 0x4DBF ) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900  && cp <= 0xFAFF ) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

uint32_t to_lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp >= 0xFF21 && cp <= 0xFF3A) return cp + 32;
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) return cp + 32;   // Latin-1 Supplement
    if (cp == 0x178) return 0xFF;                                 // Y with diaeresis
    if (cp == 0x130) return 'i';                                  // I with dot above
    // Latin Extended-A alternates upper/lower. Two runs start on an even
    // codepoint (A-with-macron at 0x100), one on an odd (L-with-acute at 0x139).
    if ((cp >= 0x100 && cp <= 0x137) || (cp >= 0x14A && cp <= 0x177))
        return (cp % 2 == 0) ? cp + 1 : cp;
    if ((cp >= 0x139 && cp <= 0x148) || (cp >= 0x179 && cp <= 0x17E))
        return (cp % 2 == 1) ? cp + 1 : cp;
    return cp;
}

// NFD + drop category Mn, for the lowercase Latin letters that survive to_lower.
// 0 means "the codepoint has no decomposition, keep it"; a returned combining
// mark can never appear because those are dropped by the caller.
char accent_base(uint32_t cp) {
    if (cp >= 0xE0 && cp <= 0xFF) {
        static const char t[32] = {
            //  e0   e1   e2   e3   e4   e5   e6(ae) e7
                'a', 'a', 'a', 'a', 'a', 'a', 0,   'c',
            //  e8   e9   ea   eb   ec   ed   ee   ef
                'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',
            //  f0(eth) f1  f2   f3   f4   f5   f6   f7(div)
                0,   'n', 'o', 'o', 'o', 'o', 'o', 0,
            //  f8(oslash) f9 fa  fb   fc   fd   fe(thorn) ff
                0,   'u', 'u', 'u', 'u', 'y', 0,   'y',
        };
        return t[cp - 0xE0];
    }
    if (cp >= 0x100 && cp <= 0x17F) {
        // One entry per codepoint; 0 for the letters NFD leaves alone
        // (d-stroke, h-bar, dotless i, ij, kra, l-stroke, eng, oe, t-bar, long s).
        static const char t[128] = {
            'A','a','A','a','A','a','C','c','C','c','C','c','C','c','D','d',  // 100-10f
             0 , 0 ,'E','e','E','e','E','e','E','e','E','e','G','g','G','g',  // 110-11f
            'G','g','G','g','H','h', 0 , 0 ,'I','i','I','i','I','i','I','i',  // 120-12f
            'I', 0 , 0 , 0 ,'J','j','K','k', 0 ,'L','l','L','l','L','l', 0 ,  // 130-13f
             0 , 0 , 0 ,'N','n','N','n','N','n', 0 , 0 , 0 ,'O','o','O','o',  // 140-14f
            'O','o', 0 , 0 ,'R','r','R','r','R','r','S','s','S','s','S','s',  // 150-15f
            'S','s','T','t','T','t', 0 , 0 ,'U','u','U','u','U','u','U','u',  // 160-16f
            'U','u','U','u','W','w','Y','y','Y','Z','z','Z','z','Z','z', 0 ,  // 170-17f
        };
        const char base = t[cp - 0x100];
        return (base >= 'A' && base <= 'Z') ? (char)(base + 32) : base;
    }
    return 0;
}

bool is_combining_mark(uint32_t cp) {
    static const uint32_t r[][2] = {
        {0x300, 0x36F}, {0x180B, 0x180D}, {0x1AB0, 0x1ABD}, {0x1DC0, 0x1DF5}, {0x1DFC, 0x1DFF},
        {0x20D0, 0x20DC}, {0x20E1, 0x20E1}, {0x20E5, 0x20F0}, {0x302A, 0x302D}, {0x3099, 0x309A},
        {0xFE00, 0xFE0F}, {0xFE20, 0xFE2F}, {0xE0100, 0xE01EF},
    };
    return cp >= 0x300 && in_ranges(cp, r);
}

} // namespace

// ---------------------------------------------------------------- load ------
bool BertTokenizer::load(const std::string& dir) { return load_file(dir + "/vocab.txt"); }

bool BertTokenizer::load_file(const std::string& vocab_path) {
    std::ifstream f(vocab_path);
    if (!f) {
        std::fprintf(stderr, "bert tokenizer: cannot open %s\n", vocab_path.c_str());
        return false;
    }
    vocab.clear();
    std::string line;
    int id = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        vocab.emplace(line, id++);
    }
    if (vocab.empty()) {
        std::fprintf(stderr, "bert tokenizer: %s is empty\n", vocab_path.c_str());
        return false;
    }
    auto pick = [&](const char* tok, int& out) {
        auto it = vocab.find(tok);
        if (it == vocab.end()) return false;
        out = it->second;
        return true;
    };
    if (!pick("[CLS]", cls_id) || !pick("[SEP]", sep_id) ||
        !pick("[PAD]", pad_id) || !pick("[UNK]", unk_id)) {
        std::fprintf(stderr, "bert tokenizer: %s is missing [CLS]/[SEP]/[PAD]/[UNK]\n",
                     vocab_path.c_str());
        return false;
    }
    return true;
}

// ------------------------------------------------------------- encode -------
std::vector<int> BertTokenizer::pieces(const std::string& text) const {
    // BasicTokenizer: clean, pad CJK, then split on whitespace.
    static const char* const specials[] = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
    const uint32_t special = 0x110000;
    std::vector<uint32_t> cps;
    cps.reserve(text.size());
    auto clean = [&](const std::string& part) {
        for (uint32_t cp : utf8_decode(part)) {
            if (cp == 0xFFFD || is_control(cp)) continue;
            if (is_whitespace(cp)) { cps.push_back(' '); continue; }
            if (is_cjk(cp)) { cps.push_back(' '); cps.push_back(cp); cps.push_back(' '); continue; }
            cps.push_back(cp);
        }
    };
    size_t done = 0;
    for (size_t i = text.find('['); i != std::string::npos; i = text.find('[', i + 1)) {
        for (const char* sp : specials) {
            const size_t n = std::strlen(sp);
            if (text.compare(i, n, sp) != 0) continue;
            auto it = vocab.find(sp);
            if (it == vocab.end()) continue;
            clean(text.substr(done, i - done));
            cps.push_back(special + (uint32_t)it->second);
            done = i + n;
            break;
        }
    }
    clean(text.substr(done));

    std::vector<int> ids;
    std::vector<std::string> words;      // one word, then its punctuation splits
    std::vector<uint32_t> word;
    auto flush_word = [&]() {
        if (word.empty()) return;
        // lowercase + strip accents, then split off punctuation
        words.clear();
        std::string cur;
        for (uint32_t raw : word) {
            uint32_t cp = lower_case ? to_lower(raw) : raw;
            if (lower_case) {
                if (is_combining_mark(cp)) continue;
                if (cp >= 0xAC00 && cp <= 0xD7A3) {
                    const uint32_t s = cp - 0xAC00;
                    utf8_append(cur, 0x1100 + s / 588);
                    utf8_append(cur, 0x1161 + s % 588 / 28);
                    if (s % 28) utf8_append(cur, 0x11A7 + s % 28);
                    continue;
                }
                const char base = accent_base(cp);
                if (base) cp = (uint32_t)(unsigned char)base;
            }
            if (is_punctuation(cp)) {
                if (!cur.empty()) { words.push_back(cur); cur.clear(); }
                std::string p;
                utf8_append(p, cp);
                words.push_back(p);
            } else {
                utf8_append(cur, cp);
            }
        }
        if (!cur.empty()) words.push_back(cur);
        word.clear();

        // WordpieceTokenizer: greedy longest-match-first, "##" on continuations.
        for (const std::string& w : words) {
            const std::vector<uint32_t> wc = utf8_decode(w);
            if ((int)wc.size() > max_chars) { ids.push_back(unk_id); continue; }
            std::vector<int> sub;
            size_t start = 0;
            bool ok = true;
            while (start < wc.size()) {
                size_t end = wc.size();
                int found = -1;
                while (end > start) {
                    std::string piece = start > 0 ? "##" : "";
                    for (size_t k = start; k < end; k++) utf8_append(piece, wc[k]);
                    auto it = vocab.find(piece);
                    if (it != vocab.end()) { found = it->second; break; }
                    end--;
                }
                if (found < 0) { ok = false; break; }
                sub.push_back(found);
                start = end;
            }
            if (ok) ids.insert(ids.end(), sub.begin(), sub.end());
            else    ids.push_back(unk_id);
        }
    };

    for (uint32_t cp : cps) {
        if (cp == ' ') flush_word();
        else if (cp >= special) { flush_word(); ids.push_back((int)(cp - special)); }
        else word.push_back(cp);
    }
    flush_word();
    return ids;
}

std::vector<int> BertTokenizer::encode(const std::string& text, int max_len, bool pad,
                                       int* n_real) const {
    std::vector<int> ids = pieces(text);
    // truncation=True: the body is cut so [CLS] body [SEP] fits in max_len.
    const int body = max_len >= 2 ? max_len - 2 : 0;
    if ((int)ids.size() > body) ids.resize((size_t)body);

    std::vector<int> out;
    out.reserve((size_t)max_len);
    out.push_back(cls_id);
    out.insert(out.end(), ids.begin(), ids.end());
    out.push_back(sep_id);
    if (n_real) *n_real = (int)out.size();
    if (pad) while ((int)out.size() < max_len) out.push_back(pad_id);
    return out;
}

} // namespace tcpu
