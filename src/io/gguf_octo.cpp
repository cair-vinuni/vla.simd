/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Octo: tools/convert_octo.py fed from vla.cpp's GGUF. vla.cpp converts through
// octo-pytorch, so the tensors arrive in PyTorch layout: Linear [out, in], conv
// [Cout, Cin, k, k] (raw, standardized at runtime), attention q/k/v packed in
// one in_proj. This adapter redoes what convert_octo.py does from the JAX tree:
// fold the weight standardization, lay convs out [Cout, k, k, Cin], split qkv,
// and derive the beta schedule. The GGUF also embeds T5's sentencepiece model,
// which becomes tok/vocab.txt, and every dataset's action statistics, picked
// the way vla.cpp picks them: $VLA_OCTO_UNNORM_DATASET, else config.txt's
// `dataset`, else the only one, else bridge_dataset.

#include "io/gguf_models.h"
#include "io/json.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace tcpu {
namespace io {

namespace {

int count(const Gguf& g, const std::string& prefix, const std::string& suffix) {
    int n = 0;
    while (g.tensor(prefix + std::to_string(n) + suffix)) n++;
    return n;
}

// sentencepiece ModelProto -> (piece, score) per id. Only field 1 (pieces) is
// read; inside a piece, field 1 is the string and field 2 the float score.
bool spm_pieces(const std::vector<uint8_t>& b, std::vector<std::pair<std::string, float>>& out) {
    auto varint = [](const uint8_t*& p, const uint8_t* e, uint64_t& v) {
        v = 0;
        for (int sh = 0; p < e && sh < 64; sh += 7) {
            const uint8_t c = *p++;
            v |= (uint64_t)(c & 0x7F) << sh;
            if (!(c & 0x80)) return true;
        }
        return false;
    };
    auto skip = [&](const uint8_t*& p, const uint8_t* e, uint32_t wt) {
        uint64_t v;
        if (wt == 0) return varint(p, e, v);
        if (wt == 1) { if (e - p < 8) return false; p += 8; return true; }
        if (wt == 5) { if (e - p < 4) return false; p += 4; return true; }
        if (wt == 2) {
            if (!varint(p, e, v) || v > (uint64_t)(e - p)) return false;
            p += v;
            return true;
        }
        return false;
    };
    const uint8_t* p = b.data();
    const uint8_t* e = p + b.size();
    while (p < e) {
        uint64_t tag;
        if (!varint(p, e, tag)) return false;
        const uint32_t field = (uint32_t)(tag >> 3), wt = (uint32_t)(tag & 7);
        if (field != 1 || wt != 2) {
            if (!skip(p, e, wt)) return false;
            continue;
        }
        uint64_t len;
        if (!varint(p, e, len) || len > (uint64_t)(e - p)) return false;
        const uint8_t* q = p;
        const uint8_t* qe = p + len;
        p = qe;
        std::string piece;
        float score = 0.0f;
        while (q < qe) {
            uint64_t t2;
            if (!varint(q, qe, t2)) return false;
            const uint32_t f2 = (uint32_t)(t2 >> 3), w2 = (uint32_t)(t2 & 7);
            if (f2 == 1 && w2 == 2) {
                uint64_t n;
                if (!varint(q, qe, n) || n > (uint64_t)(qe - q)) return false;
                piece.assign((const char*)q, (size_t)n);
                q += n;
            } else if (f2 == 2 && w2 == 5) {
                if (qe - q < 4) return false;
                std::memcpy(&score, q, 4);
                q += 4;
            } else if (!skip(q, qe, w2)) {
                return false;
            }
        }
        out.emplace_back(std::move(piece), score);
    }
    return !out.empty();
}

// convert_octo.py's weight_standardize over one conv, in float64:
// per output channel, (w - mean) / (std + 1e-5) over (kh, kw, cin).
// In: PyTorch [Cout, Cin, k, k]. Out: the engine's [Cout, k, k, Cin].
std::vector<float> std_conv(const std::vector<float>& w, int64_t cout, int64_t cin, int64_t k) {
    const int64_t n = cin * k * k;
    std::vector<float> out((size_t)(cout * n));
    std::vector<double> v((size_t)n);
    for (int64_t o = 0; o < cout; o++) {
        for (int64_t i = 0; i < n; i++) v[(size_t)i] = w[(size_t)(o * n + i)];
        double m = 0;
        for (double x : v) m += x;
        m /= (double)n;
        double var = 0;
        for (double x : v) var += (x - m) * (x - m);
        const double sd = std::sqrt(var / (double)n);
        for (int64_t ci = 0; ci < cin; ci++)
            for (int64_t kh = 0; kh < k; kh++)
                for (int64_t kw = 0; kw < k; kw++)
                    out[(size_t)(((o * k + kh) * k + kw) * cin + ci)] =
                        (float)((v[(size_t)((ci * k + kh) * k + kw)] - m) / (sd + 1e-5));
    }
    return out;
}

void split_qkv(Blob& b, const std::vector<float>& w, const std::vector<float>& bias, int64_t D) {
    for (int i = 0; i < 3; i++) {
        b.f32(w.data() + (size_t)(i * D * D), (size_t)(D * D));
        b.f32(bias.data() + (size_t)(i * D), (size_t)D);
    }
}

} // namespace

bool adapt_octo(const Gguf& g, const Sidecar& side, Files& out, std::string& err) {
    TensorReader t{g, err};
    std::map<std::string, std::string> cfg = side.config();

    if (g.str("octo.action.head_type") != "diffusion") {
        err = g.path() + ": Octo action head '" + g.str("octo.action.head_type") +
              "' is not supported (diffusion is)";
        return false;
    }
    if (g.tensor("octo.obs.proprio.proj.weight")) {
        err = g.path() + ": Octo with a proprio tokenizer is not supported";
        return false;
    }

    // ---- T5 encoder ----
    const std::vector<int64_t> emb = t.shape("octo.t5.tok_embd.weight");
    const std::vector<int64_t> rel = t.shape("octo.t5.blk.0.attn_rel_b.weight");
    const std::vector<int64_t> wi = t.shape("octo.t5.blk.0.ffn_up.weight");
    const int T5L = count(g, "octo.t5.blk.", ".attn_q.weight");
    if (!t.ok()) return false;
    const int64_t TD = emb[1], TFF = wi[0], NB = rel[0], TNH = rel[1];
    const int64_t n_lang = t.u32("octo.tokens.language");

    Meta t5_meta;
    t5_meta.i("d_model", TD).i("n_layers", T5L).i("n_heads", TNH).i("d_kv", TD / TNH).i("d_ff", TFF)
           .i("vocab", emb[0]).i("n_buckets", NB).i("max_dist", 128).f("eps", 1e-6).i("n_tokens", n_lang);
    Blob t5;
    t5.f32(t.f32("octo.t5.tok_embd.weight", {emb[0], TD}));
    t5.f32(t.f32("octo.t5.blk.0.attn_rel_b.weight", {NB, TNH}));
    for (int L = 0; L < T5L; L++) {
        const std::string p = "octo.t5.blk." + std::to_string(L) + ".";
        t5.f32(t.f32(p + "attn_norm.weight", {TD}));
        for (const char* n : {"attn_q", "attn_k", "attn_v", "attn_o"})
            t5.f32(t.f32(p + n + ".weight", {TD, TD}));
        t5.f32(t.f32(p + "ffn_norm.weight", {TD}));
        t5.f32(t.f32(p + "ffn_up.weight", {TFF, TD}));
        t5.f32(t.f32(p + "ffn_down.weight", {TD, TFF}));
    }
    t5.f32(t.f32("octo.t5.output_norm.weight", {TD}));
    if (!t.ok()) return false;

    // ---- SmallStem16 x2 ----
    auto stem = [&](const char* view, Blob& b, Meta& m) {
        const std::string p = std::string("octo.obs.") + view + ".";
        const int NL = count(g, p + "stem.", ".conv.weight");
        std::vector<int64_t> feats;
        int64_t in_ch = 0, k = 0;
        for (int i = 0; i < NL && t.ok(); i++) {
            const std::string c = p + "stem." + std::to_string(i) + ".";
            const std::vector<int64_t> s = t.shape(c + "conv.weight");      // [Cout, Cin, k, k]
            if (!t.ok()) return;
            if (i == 0) { in_ch = s[1]; k = s[2]; }
            feats.push_back(s[0]);
            b.f32(std_conv(t.f32(c + "conv.weight", {s[0], s[1], s[2], s[3]}), s[0], s[1], s[2]));
            b.f32(t.f32(c + "conv.bias", {s[0]}));
            b.f32(t.f32(c + "gn.weight", {s[0]}));
            b.f32(t.f32(c + "gn.bias", {s[0]}));
        }
        const std::vector<int64_t> pe = t.shape(p + "patch_embd.weight");   // [E, C, 1, 1]
        if (!t.ok()) return;
        b.f32(t.f32(p + "patch_embd.weight", {pe[0], pe[1], 1, 1}));        // == [E, C]
        b.f32(t.f32(p + "patch_embd.bias", {pe[0]}));
        std::string f;
        for (int64_t x : feats) f += (f.empty() ? "" : " ") + std::to_string(x);
        m.i("in_ch", in_ch).i("n_layers", NL).i("k", k).i("stride", 2).i("pad", 1).s("features", f)
         .i("embed_dim", pe[0]).i("gn_groups", 32).f("gn_eps", 1e-6);
    };
    Blob stem_p, stem_w;
    Meta stem_p_meta, stem_w_meta;
    stem("primary", stem_p, stem_p_meta);
    stem("wrist", stem_w, stem_w_meta);
    if (!t.ok()) return false;

    // ---- block transformer ----
    const int64_t D = t.u32("octo.embedding_length"), NH = t.u32("octo.attention.head_count");
    const int64_t MLP = t.u32("octo.feed_forward_length");
    const int64_t tok_p = t.u32("octo.tokens.primary"), tok_w = t.u32("octo.tokens.wrist");
    const int64_t n_readout = t.u32("octo.readout.count");
    const int64_t window = t.u32("octo.window_size");
    const int BL = count(g, "octo.blk.", ".attn_qkv.weight");
    const std::vector<int64_t> pp = t.shape("octo.obs.primary.pos_embd");   // [horizon, tok, D]
    const std::vector<int64_t> sp = t.shape("octo.obs.primary.proj.weight");
    if (!t.ok()) return false;
    const int64_t HMAX = pp[0], SD = sp[1];

    Meta octo_meta;
    octo_meta.i("d", D).i("n_layers", BL).i("heads", NH).i("head_dim", D / NH).i("mlp", MLP)
             .i("max_horizon", HMAX).i("n_task", n_lang).i("tok_primary", tok_p).i("tok_wrist", tok_w)
             .i("n_readout", n_readout).i("t5_dim", TD).i("stem_dim", SD)
             .f("ln_eps", t.num("octo.attention.layer_norm_eps")).i("window", window);
    Blob octo;
    octo.f32(t.f32("octo.task.language.proj.weight", {D, TD}));
    octo.f32(t.f32("octo.task.language.proj.bias", {D}));
    octo.f32(t.f32("octo.obs.primary.proj.weight", {D, SD}));
    octo.f32(t.f32("octo.obs.primary.proj.bias", {D}));
    octo.f32(t.f32("octo.obs.wrist.proj.weight", {D, SD}));
    octo.f32(t.f32("octo.obs.wrist.proj.bias", {D}));
    octo.f32(t.f32("octo.task.language.pos_embd", {1, n_lang, D}));
    octo.f32(t.f32("octo.obs.primary.pos_embd", {1, HMAX, tok_p, D}));
    octo.f32(t.f32("octo.obs.wrist.pos_embd", {1, HMAX, tok_w, D}));
    octo.f32(t.f32("octo.readout.action.pos_embd", {1, HMAX, n_readout, D}));
    for (int L = 0; L < BL; L++) {
        const std::string p = "octo.blk." + std::to_string(L) + ".";
        octo.f32(t.f32(p + "attn_norm.weight", {D}));
        octo.f32(t.f32(p + "attn_norm.bias", {D}));
        split_qkv(octo, t.f32(p + "attn_qkv.weight", {3 * D, D}), t.f32(p + "attn_qkv.bias", {3 * D}), D);
        octo.f32(t.f32(p + "attn_o.weight", {D, D}));
        octo.f32(t.f32(p + "attn_o.bias", {D}));
        octo.f32(t.f32(p + "ffn_norm.weight", {D}));
        octo.f32(t.f32(p + "ffn_norm.bias", {D}));
        octo.f32(t.f32(p + "ffn_up.weight", {MLP, D}));
        octo.f32(t.f32(p + "ffn_up.bias", {MLP}));
        octo.f32(t.f32(p + "ffn_down.weight", {D, MLP}));
        octo.f32(t.f32(p + "ffn_down.bias", {D}));
    }
    octo.f32(t.f32("octo.output_norm.weight", {D}));
    octo.f32(t.f32("octo.output_norm.bias", {D}));
    if (!t.ok()) return false;

    // ---- diffusion head + beta schedule ----
    const int64_t A = t.u32("octo.action.dim"), AH = t.u32("octo.action.horizon");
    const int64_t steps = t.u32("octo.diffusion.steps"), time_dim = t.u32("octo.diffusion.time_dim");
    const int64_t hidden = t.u32("octo.diffusion.hidden"), n_blocks = t.u32("octo.diffusion.num_blocks");
    const double max_action = t.num("octo.diffusion.max_action"), s = t.num("octo.diffusion.s");
    if (!t.ok()) return false;
    if (g.str("octo.diffusion.beta_schedule") != "cosine") {
        err = g.path() + ": beta schedule '" + g.str("octo.diffusion.beta_schedule") +
              "' is not supported (cosine is)";
        return false;
    }
    const int64_t flat = A * AH, IN = time_dim + D + flat;

    Meta head_meta;
    head_meta.i("emb", D).i("action_dim", A).i("horizon", AH).i("time_dim", time_dim)
             .i("num_blocks", n_blocks).i("hidden", hidden).i("steps", steps).f("max_action", max_action);
    Blob head;
    head.f32(t.f32("octo.head.diffusion.time_fourier.weight", {time_dim / 2, 1}));
    head.f32(t.f32("octo.head.diffusion.cond.0.weight", {2 * time_dim, time_dim}));
    head.f32(t.f32("octo.head.diffusion.cond.0.bias", {2 * time_dim}));
    head.f32(t.f32("octo.head.diffusion.cond.1.weight", {time_dim, 2 * time_dim}));
    head.f32(t.f32("octo.head.diffusion.cond.1.bias", {time_dim}));
    head.f32(t.f32("octo.head.diffusion.reverse.in.weight", {hidden, IN}));
    head.f32(t.f32("octo.head.diffusion.reverse.in.bias", {hidden}));
    for (int64_t i = 0; i < n_blocks; i++) {
        const std::string p = "octo.head.diffusion.reverse.blk." + std::to_string(i) + ".";
        head.f32(t.f32(p + "ln.weight", {hidden}));
        head.f32(t.f32(p + "ln.bias", {hidden}));
        head.f32(t.f32(p + "fc1.weight", {4 * hidden, hidden}));
        head.f32(t.f32(p + "fc1.bias", {4 * hidden}));
        head.f32(t.f32(p + "fc2.weight", {hidden, 4 * hidden}));
        head.f32(t.f32(p + "fc2.bias", {hidden}));
    }
    head.f32(t.f32("octo.head.diffusion.reverse.out.weight", {flat, hidden}));
    head.f32(t.f32("octo.head.diffusion.reverse.out.bias", {flat}));
    if (!t.ok()) return false;
    {   // cosine_beta_schedule(), then float32 alphas and their running product
        std::vector<double> ac((size_t)steps + 1);
        const double pi = 3.14159265358979323846;
        for (int64_t i = 0; i <= steps; i++) {
            const double x = std::cos(((double)i / (double)steps + s) / (1 + s) * pi * 0.5);
            ac[(size_t)i] = x * x;
        }
        for (int64_t i = steps; i >= 0; i--) ac[(size_t)i] /= ac[0];
        std::vector<float> betas((size_t)steps), alphas((size_t)steps), alpha_hats((size_t)steps);
        float run = 1.0f;
        for (int64_t i = 0; i < steps; i++) {
            const double b = std::min(std::max(1 - ac[(size_t)i + 1] / ac[(size_t)i], 0.0), 0.999);
            betas[(size_t)i] = (float)b;
            alphas[(size_t)i] = 1.0f - betas[(size_t)i];
            run = i == 0 ? alphas[0] : run * alphas[(size_t)i];
            alpha_hats[(size_t)i] = run;
        }
        head.f32(betas);
        head.f32(alphas);
        head.f32(alpha_hats);
    }

    // ---- action statistics ----
    Json stats;
    if (!Json::parse(g.str("octo.dataset_statistics", "{}"), stats) || stats.kind != Json::Object) {
        err = g.path() + ": octo.dataset_statistics is not a JSON object";
        return false;
    }
    const Json* block = nullptr;
    std::string key;
    if (const Json* a = stats.at("action"); a && a->at("mean")) {
        block = &stats;                                  // single-dataset checkpoint, written flat
    } else {
        const char* env = std::getenv("VLA_OCTO_UNNORM_DATASET");
        if (env && *env) key = env;
        else if (cfg.count("dataset")) key = cfg["dataset"];
        else if (stats.obj.size() == 1) key = stats.obj.begin()->first;
        else if (stats.at("bridge_dataset")) key = "bridge_dataset";
        else {
            std::string names;
            for (const auto& [k, v] : stats.obj) names += (names.empty() ? "" : ", ") + k;
            err = g.path() + ": " + std::to_string(stats.obj.size()) + " datasets to un-normalize "
                  "against (" + names + "); set VLA_OCTO_UNNORM_DATASET or `dataset` in config.txt";
            return false;
        }
        block = stats.at(key);
        if (!block) { err = g.path() + ": dataset_statistics has no '" + key + "'"; return false; }
    }
    const Json* act = block->at("action");
    Blob s_mean, s_std, s_mask;
    for (const char* name : {"mean", "std", "mask"}) {
        const Json* v = act ? act->at(name) : nullptr;
        if (!v || v->kind != Json::Array || (int64_t)v->arr.size() != A) {
            err = g.path() + ": action " + name + " statistics missing or not " + std::to_string(A) + "-wide";
            return false;
        }
        std::vector<float> f;
        for (const Json& x : v->arr) f.push_back((float)x.as_num());
        (name[1] == 'e' ? s_mean : name[1] == 't' ? s_std : s_mask).f32(f);
    }
    if (!key.empty()) std::fprintf(stderr, "vla-simd: octo un-normalizes against %s\n", key.c_str());

    // ---- tokenizer ----
    const GgufValue* spm = g.get("octo.tokenizer.spm_model");
    std::vector<std::pair<std::string, float>> pieces;
    if (!spm || spm->bytes.empty() || !spm_pieces(spm->bytes, pieces)) {
        err = g.path() + ": octo.tokenizer.spm_model missing or not a sentencepiece model";
        return false;
    }
    std::string vocab;
    for (const auto& [piece, score] : pieces) vocab += piece + "\t" + pyrepr(score) + "\n";
    // T5's 100 sentinels follow the sentencepiece vocabulary, highest id first
    for (int i = 99; i >= 0; i--) vocab += "<extra_id_" + std::to_string(i) + ">\t0.0\n";

    std::string config_txt = config_with(side, {
        {"window", std::to_string(window)}, {"steps", std::to_string(steps)}});
    if (!key.empty() && !cfg.count("dataset")) config_txt += "dataset " + key + "\n";

    out["t5.meta"] = t5_meta.text;                   out["t5.bin"] = std::move(t5.bytes);
    out["stem_primary.meta"] = stem_p_meta.text;     out["stem_primary.bin"] = std::move(stem_p.bytes);
    out["stem_wrist.meta"] = stem_w_meta.text;       out["stem_wrist.bin"] = std::move(stem_w.bytes);
    out["octo.meta"] = octo_meta.text;               out["octo.bin"] = std::move(octo.bytes);
    out["head.meta"] = head_meta.text;               out["head.bin"] = std::move(head.bytes);
    out["stats_action_mean.bin"] = std::move(s_mean.bytes);
    out["stats_action_std.bin"] = std::move(s_std.bytes);
    out["stats_action_mask.bin"] = std::move(s_mask.bytes);
    out["tok/vocab.txt"] = vocab;
    out["config.txt"] = config_txt;
    return true;
}

} // namespace io
} // namespace tcpu
