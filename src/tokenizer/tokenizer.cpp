/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <regex>

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

// Llama-3 / GPT-4 pretokenizer regex, ASCII substitution (\p{L}->A-Za-z, \p{N}->0-9).
static const std::regex& pat() {
    static const std::regex p(
        "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])"
        "|[^\\r\\nA-Za-z0-9]?[A-Za-z]+"
        "|[0-9]{1,3}"
        "| ?[^\\sA-Za-z0-9]+[\\r\\n]*"
        "|\\s*[\\r\\n]+"
        "|\\s+(?!\\S)"
        "|\\s+",
        std::regex::ECMAScript);
    return p;
}

bool Tokenizer::load(const std::string& dir) {
    build_byte2str(byte2str);

    std::ifstream vf(dir + "/vocab.txt");
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

    std::ifstream mf(dir + "/merges.txt");
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

    std::ifstream sf(dir + "/specials.txt");
    if (sf) {
        while (std::getline(sf, line)) {
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            specials[line.substr(tab+1)] = std::stoi(line.substr(0, tab));
        }
    }
    return !vocab.empty() && !ranks.empty();
}

std::vector<int> Tokenizer::bpe_encode(const std::string& piece_bytes) const {
    std::vector<std::string> word;
    for (unsigned char b : piece_bytes)
        word.push_back(byte2str[b]);

    // ignore_merges: whole byte-level piece directly in vocab -> emit it (tiktoken)
    std::string joined;
    for (auto& s : word)
        joined += s;

    auto jit = vocab.find(joined);
    if (jit != vocab.end()) return {jit->second};

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
    for (auto it = std::sregex_iterator(text.begin(), text.end(), pat()); it != std::sregex_iterator(); ++it) {
        auto sub = bpe_encode(it->str());
        ids.insert(ids.end(), sub.begin(), sub.end());
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos) ids.push_back(bos_id);
    encode_text(text, ids);
    return ids;
}

std::vector<int> Tokenizer::encode_with_specials(const std::string& text, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos) ids.push_back(bos_id);

    std::string buf;
    for (size_t i=0; i<text.size();) {
        if (text[i] == '<') {
            // longest special-token match at this position
            int best_id     = -1;
            size_t best_len = 0;
            for (const auto& kv : specials) {
                const std::string& c = kv.first;
                if (c.size() > best_len && i+c.size() <= text.size() && text.compare(i, c.size(), c) == 0) {
                    best_len = c.size();
                    best_id  = kv.second;
                }
            }
            if (best_id >= 0) {
                if (!buf.empty()) {
                    encode_text(buf, ids);
                    buf.clear();
                }
                ids.push_back(best_id);
                i += best_len;
                continue;
            }
        }
        buf += text[i++];
    }

    if (!buf.empty()) encode_text(buf, ids);
    return ids;
}

} // namespace tcpu
