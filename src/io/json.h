/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Minimal JSON reader for the configuration blobs vla.cpp stores as GGUF
// strings (TurboVLA's config_json, Octo's dataset_statistics). Numbers are
// doubles, which is what Python's json module produced them from.

namespace tcpu {
namespace io {

struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    const Json* at(const std::string& k) const {
        if (kind != Object) return nullptr;
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
    bool is_num() const { return kind == Number || kind == Bool; }
    double as_num() const { return kind == Bool ? (b ? 1.0 : 0.0) : num; }

    static bool parse(const std::string& s, Json& out) {
        size_t i = 0;
        if (!value(s, i, out, 0)) return false;
        ws(s, i);
        return i == s.size();
    }

private:
    static void ws(const std::string& s, size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }
    static void utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | cp >> 6); o += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            o += (char)(0xE0 | cp >> 12); o += (char)(0x80 | ((cp >> 6) & 0x3F));
            o += (char)(0x80 | (cp & 0x3F));
        } else {
            o += (char)(0xF0 | cp >> 18); o += (char)(0x80 | ((cp >> 12) & 0x3F));
            o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F));
        }
    }
    static bool hex4(const std::string& s, size_t& i, unsigned& cp) {
        if (i + 4 > s.size()) return false;
        cp = (unsigned)std::strtoul(s.substr(i, 4).c_str(), nullptr, 16);
        i += 4;
        return true;
    }
    static bool string(const std::string& s, size_t& i, std::string& o) {
        if (s[i] != '"') return false;
        i++;
        while (i < s.size() && s[i] != '"') {
            if (s[i] != '\\') { o += s[i++]; continue; }
            if (++i >= s.size()) return false;
            const char e = s[i++];
            switch (e) {
                case '"': o += '"'; break;   case '\\': o += '\\'; break;
                case '/': o += '/'; break;   case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;  case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;  case 't': o += '\t'; break;
                case 'u': {
                    unsigned cp;
                    if (!hex4(s, i, cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i+1] == 'u') {
                        size_t j = i + 2;
                        unsigned lo;
                        if (hex4(s, j, lo) && lo >= 0xDC00 && lo < 0xE000) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i = j;
                        }
                    }
                    utf8(o, cp);
                    break;
                }
                default: return false;
            }
        }
        if (i >= s.size()) return false;
        i++;
        return true;
    }
    static bool value(const std::string& s, size_t& i, Json& v, int depth) {
        if (depth > 64) return false;
        ws(s, i);
        if (i >= s.size()) return false;
        const char c = s[i];
        if (c == '{') {
            v.kind = Object;
            i++;
            ws(s, i);
            if (i < s.size() && s[i] == '}') { i++; return true; }
            for (;;) {
                ws(s, i);
                std::string k;
                if (i >= s.size() || !string(s, i, k)) return false;
                ws(s, i);
                if (i >= s.size() || s[i++] != ':') return false;
                if (!value(s, i, v.obj[k], depth + 1)) return false;
                ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == ',') { i++; continue; }
                if (s[i] == '}') { i++; return true; }
                return false;
            }
        }
        if (c == '[') {
            v.kind = Array;
            i++;
            ws(s, i);
            if (i < s.size() && s[i] == ']') { i++; return true; }
            for (;;) {
                v.arr.emplace_back();
                if (!value(s, i, v.arr.back(), depth + 1)) return false;
                ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == ',') { i++; continue; }
                if (s[i] == ']') { i++; return true; }
                return false;
            }
        }
        if (c == '"') { v.kind = String; return string(s, i, v.str); }
        if (s.compare(i, 4, "true") == 0)  { v.kind = Bool; v.b = true;  i += 4; return true; }
        if (s.compare(i, 5, "false") == 0) { v.kind = Bool; v.b = false; i += 5; return true; }
        if (s.compare(i, 4, "null") == 0)  { v.kind = Null; i += 4; return true; }
        // Python's json.dumps writes NaN/Infinity for non-finite floats
        if (s.compare(i, 3, "NaN") == 0)   { v.kind = Number; v.num = std::strtod("nan", nullptr); i += 3; return true; }
        if (s.compare(i, 8, "Infinity") == 0)  { v.kind = Number; v.num = HUGE_VAL;  i += 8; return true; }
        if (s.compare(i, 9, "-Infinity") == 0) { v.kind = Number; v.num = -HUGE_VAL; i += 9; return true; }
        char* end = nullptr;
        v.num = std::strtod(s.c_str() + i, &end);
        if (end == s.c_str() + i) return false;
        v.kind = Number;
        i = (size_t)(end - s.c_str());
        return true;
    }
};

} // namespace io
} // namespace tcpu
