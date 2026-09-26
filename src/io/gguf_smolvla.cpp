/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// SmolVLA: vla.cpp's convert_smolvla_to_gguf.py renames the lerobot tensors
// without touching their values, so this is tools/convert_hf_safetensors.py
// with the names swapped. Two things the GGUF does not record come from the
// sidecar config.txt: the SigLIP position-id mode (pos_ids, default identity,
// which is what vla.cpp runs) and the camera count (n_views, default 2).

#include "io/gguf_models.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace tcpu {
namespace io {

namespace {

int count_layers(const Gguf& g, const std::string& prefix) {
    int n = 0;
    while (g.tensor(prefix + std::to_string(n) + ".attn_q.weight")) n++;
    return n;
}

// SmolVLMVisionEmbeddings' patch -> position row, as position_ids() in
// tools/convert_hf_safetensors.py.
std::vector<int64_t> position_ids(int side, bool shifted) {
    std::vector<int64_t> bucket((size_t)side);
    for (int k = 0; k < side; k++) {
        double coord = (double)k / side;
        if (shifted) coord *= (1 - 1e-6);
        coord = (double)(float)coord;
        int b = 0;                           // searchsorted(side="right")
        for (int j = 1; j < side; j++)
            if ((double)j / side <= coord) b = j;
        bucket[(size_t)k] = b;
    }
    std::vector<int64_t> pos((size_t)side * side);
    for (int r = 0; r < side; r++)
        for (int c = 0; c < side; c++) pos[(size_t)r * side + c] = bucket[(size_t)r] * side + bucket[(size_t)c];
    return pos;
}

} // namespace

bool adapt_smolvla(const Gguf& g, const Sidecar& side, Files& out, std::string& err) {
    TensorReader t{g, err};
    const std::string kv = "smolvla.";
    if (!g.tensor("vit.patch_embd.weight")) {
        err = g.path() + ": no vision tower. This is vla.cpp's older two-file layout (the tower "
              "was in a separate mmproj-*.gguf); convert again with a vla.cpp that bakes it in";
        return false;
    }
    std::map<std::string, std::string> cfg = side.config();

    // ---- language model ----
    const int64_t H = t.u32(kv + "hidden"), F = t.u32(kv + "intermediate");
    const int64_t HD = t.u32(kv + "head_dim");
    const int64_t QF = t.u32(kv + "n_q_heads") * HD, KV = t.u32(kv + "n_kv_heads") * HD;
    const int NL = count_layers(g, "vlm.blk.");
    if (!t.ok()) return false;
    if (NL < 1) { err = g.path() + ": no vlm.blk.* layers"; return false; }

    Meta vlm_meta;
    vlm_meta.i("hidden", H).i("n_q", QF / HD).i("n_kv", KV / HD).i("head_dim", HD).i("ffn", F)
            .f("eps", 1e-5).f("rope_base", 10000.0).i("n_layers", NL);
    Blob vlm;
    for (int L = 0; L < NL; L++) {
        const std::string p = "vlm.blk." + std::to_string(L) + ".";
        vlm.f32(t.f32(p + "attn_norm.weight", {H}));
        vlm.f32(t.f32(p + "ffn_norm.weight", {H}));
    }
    vlm.f32(t.f32("vlm.output_norm.weight", {H}));
    for (int L = 0; L < NL; L++) {
        const std::string p = "vlm.blk." + std::to_string(L) + ".";
        vlm.bf16(t.f32(p + "attn_q.weight", {QF, H}));
        vlm.bf16(t.f32(p + "attn_k.weight", {KV, H}));
        vlm.bf16(t.f32(p + "attn_v.weight", {KV, H}));
        vlm.bf16(t.f32(p + "attn_o.weight", {H, QF}));
        vlm.bf16(t.f32(p + "ffn_gate.weight", {F, H}));
        vlm.bf16(t.f32(p + "ffn_up.weight", {F, H}));
        vlm.bf16(t.f32(p + "ffn_down.weight", {H, F}));
    }
    if (!t.ok()) return false;

    // ---- SigLIP vision tower + connector ----
    const std::vector<int64_t> pe = t.shape("vit.patch_embd.weight");      // [H, 3, p, p]
    if (!t.ok()) return false;
    const int64_t VH = pe.size() == 4 ? pe[0] : 0;
    const int64_t patch = (int64_t)t.num(kv + "patch_size");
    const int64_t img = (int64_t)t.num(kv + "image_size");
    const int64_t sf = (int64_t)t.num(kv + "vit_pixel_shuffle");
    const int64_t heads = (int64_t)t.num(kv + "vit_heads");
    const int64_t VI = t.shape("vit.blk.0.fc1.weight").empty() ? 0 : t.shape("vit.blk.0.fc1.weight")[0];
    const int VL = count_layers(g, "vit.blk.");
    if (!t.ok()) return false;
    if (VH < 1 || patch < 1 || img % patch || heads < 1 || VH % heads || sf < 1 || VL < 1) {
        err = g.path() + ": SigLIP geometry does not close";
        return false;
    }
    const int64_t side_n = img / patch, NP = side_n * side_n;
    const int64_t n_img_tok = NP / (sf * sf);

    const std::string pos_mode = cfg.count("pos_ids") ? cfg["pos_ids"] : "identity";
    if (pos_mode != "identity" && pos_mode != "shifted") {
        err = side.dir + "/config.txt: pos_ids must be identity or shifted, not " + pos_mode;
        return false;
    }
    const std::vector<float> pos_table = t.f32("vit.pos_embd", {NP, VH});
    std::vector<float> pos_gathered((size_t)(NP * VH));
    if (t.ok()) {
        const std::vector<int64_t> pos = position_ids((int)side_n, pos_mode == "shifted");
        for (int64_t p = 0; p < NP; p++)
            std::copy_n(pos_table.begin() + pos[(size_t)p] * VH, VH, pos_gathered.begin() + p * VH);
    }

    Meta vit_meta;
    vit_meta.i("hidden", VH).i("n_heads", heads).i("head_dim", VH / heads).i("inter", VI)
            .i("n_layers", VL).i("patch", patch).i("img", img).i("n_patches", NP)
            .f("ln_eps", g.has(kv + "vit_ln_eps") ? t.num(kv + "vit_ln_eps") : 1e-6)
            .i("scale_factor", sf).i("mm_out", H).i("n_img_tok", n_img_tok);
    Blob vit;
    vit.f32(t.f32("vit.patch_embd.bias", {VH}));
    vit.f32(pos_gathered);
    for (int L = 0; L < VL; L++) {
        const std::string p = "vit.blk." + std::to_string(L) + ".";
        vit.f32(t.f32(p + "ln1.weight", {VH}));
        vit.f32(t.f32(p + "ln1.bias", {VH}));
        vit.f32(t.f32(p + "attn_q.bias", {VH}));
        vit.f32(t.f32(p + "attn_k.bias", {VH}));
        vit.f32(t.f32(p + "attn_v.bias", {VH}));
        vit.f32(t.f32(p + "attn_o.bias", {VH}));
        vit.f32(t.f32(p + "ln2.weight", {VH}));
        vit.f32(t.f32(p + "ln2.bias", {VH}));
        vit.f32(t.f32(p + "fc1.bias", {VI}));
        vit.f32(t.f32(p + "fc2.bias", {VH}));
    }
    vit.f32(t.f32("vit.post_ln.weight", {VH}));
    vit.f32(t.f32("vit.post_ln.bias", {VH}));
    vit.bf16(t.f32("vit.patch_embd.weight", {VH, pe[1], pe[2], pe[3]}));   // == [VH, PD]
    for (int L = 0; L < VL; L++) {
        const std::string p = "vit.blk." + std::to_string(L) + ".";
        vit.bf16(t.f32(p + "attn_q.weight", {VH, VH}));
        vit.bf16(t.f32(p + "attn_k.weight", {VH, VH}));
        vit.bf16(t.f32(p + "attn_v.weight", {VH, VH}));
        vit.bf16(t.f32(p + "attn_o.weight", {VH, VH}));
        vit.bf16(t.f32(p + "fc1.weight", {VI, VH}));
        vit.bf16(t.f32(p + "fc2.weight", {VH, VI}));
    }
    vit.bf16(t.f32("mm.fc.weight", {H, VH * sf * sf}));
    if (!t.ok()) return false;

    // ---- action expert (fp32 throughout, as the converter widens it) ----
    const int64_t EH = t.u32(kv + "expert_h"), EF = t.u32(kv + "expert_inter");
    const int64_t san = t.u32(kv + "self_attn_every_n_layers");
    const int64_t MAD = t.u32(kv + "max_action_dim"), MSD = t.u32(kv + "max_state_dim");
    const int EL = count_layers(g, "aex.blk.");
    if (!t.ok()) return false;
    if (EL != NL) {
        err = g.path() + ": vlm has " + std::to_string(NL) + " layers, expert has " + std::to_string(EL);
        return false;
    }
    if (san < 1) { err = g.path() + ": self_attn_every_n_layers must be >= 1"; return false; }

    Meta aex_meta;
    aex_meta.i("expert_h", EH).i("expert_ffn", EF).i("n_q", QF / HD).i("n_kv", KV / HD)
            .i("head_dim", HD).f("eps", 1e-5).f("rope_base", 10000.0).i("n_layers", EL)
            .i("self_attn_every_n", san).i("chunk", t.u32(kv + "chunk_size"))
            .i("num_steps", t.u32(kv + "num_steps")).i("max_action_dim", MAD)
            .f("min_period", t.num(kv + "min_period")).f("max_period", t.num(kv + "max_period"));
    Blob aex;
    for (int L = 0; L < EL; L++) {
        const std::string p = "aex.blk." + std::to_string(L) + ".";
        // even layers attend the expert stream, odd ones reproject the VLM cache
        const int64_t kin = L % san == 0 ? EH : KV;
        aex.f32(t.f32(p + "attn_norm.weight", {EH}));
        aex.f32(t.f32(p + "attn_q.weight", {QF, EH}));
        aex.f32(t.f32(p + "attn_k.weight", {KV, kin}));
        aex.f32(t.f32(p + "attn_v.weight", {KV, kin}));
        aex.f32(t.f32(p + "attn_o.weight", {EH, QF}));
        aex.f32(t.f32(p + "ffn_norm.weight", {EH}));
        aex.f32(t.f32(p + "ffn_gate.weight", {EF, EH}));
        aex.f32(t.f32(p + "ffn_up.weight", {EF, EH}));
        aex.f32(t.f32(p + "ffn_down.weight", {EH, EF}));
    }
    aex.f32(t.f32("aex.output_norm.weight", {EH}));
    aex.f32(t.f32("action_in_proj.weight", {EH, MAD}));
    aex.f32(t.f32("action_in_proj.bias", {EH}));
    aex.f32(t.f32("action_time_mlp_in.weight", {EH, 2 * EH}));
    aex.f32(t.f32("action_time_mlp_in.bias", {EH}));
    aex.f32(t.f32("action_time_mlp_out.weight", {EH, EH}));
    aex.f32(t.f32("action_time_mlp_out.bias", {EH}));
    aex.f32(t.f32("action_out_proj.weight", {MAD, EH}));
    aex.f32(t.f32("action_out_proj.bias", {MAD}));
    if (!t.ok()) return false;

    // ---- embedding, state head, statistics ----
    const std::vector<int64_t> es = t.shape("token_embd.weight");
    if (!t.ok()) return false;
    const int64_t vocab = es[0];
    Blob emb;
    emb.bf16(t.f32("token_embd.weight", {vocab, H}));
    Blob heads_bin;
    heads_bin.f32(t.f32("state_proj.weight", {H, MSD}));
    heads_bin.f32(t.f32("state_proj.bias", {H}));
    const int64_t RS = t.u32(kv + "real_state_dim"), RA = t.u32(kv + "real_action_dim");
    Blob sm, ss, am, as;
    sm.f32(t.f32("state_mean", {RS}));
    ss.f32(t.f32("state_std", {RS}));
    am.f32(t.f32("action_mean", {RA}));
    as.f32(t.f32("action_std", {RA}));
    if (!t.ok()) return false;

    const long long n_views = cfg.count("n_views") ? std::atoll(cfg["n_views"].c_str()) : 2;
    Meta heads_meta;
    heads_meta.i("vocab", vocab).i("hidden", H).i("max_state_dim", MSD).i("real_state_dim", RS)
              .i("real_action_dim", RA).i("n_views", n_views);
    if (g.has(kv + "norm_eps")) heads_meta.f("norm_eps", t.num(kv + "norm_eps"));

    // config.txt: the sidecar's lines, plus what the C API reads and the GGUF knows
    int pad_id = 2;
    std::string vocab_txt;
    if (side.read("tok/vocab.txt", vocab_txt)) {
        std::istringstream vs(vocab_txt);
        std::string line;
        while (std::getline(vs, line)) {
            const size_t tab = line.find('\t');
            if (tab != std::string::npos && line.compare(tab + 1, std::string::npos, "<|im_end|>") == 0) {
                pad_id = std::atoi(line.substr(0, tab).c_str());
                break;
            }
        }
    }
    std::string config_txt = config_with(side, {
        {"tokenizer_max_length", std::to_string(t.u32(kv + "tokenizer_max_length"))},
        {"pad_token_id", std::to_string(pad_id)},
        {"chunk", std::to_string(t.u32(kv + "chunk_size"))},
        {"num_steps", std::to_string(t.u32(kv + "num_steps"))},
        {"n_views", std::to_string(n_views)},
        {"pos_ids", pos_mode}});
    if (!t.ok()) return false;

    out["vlm.meta"] = vlm_meta.text;     out["vlm.bin"] = std::move(vlm.bytes);
    out["vit.meta"] = vit_meta.text;     out["vit.bin"] = std::move(vit.bytes);
    out["aex.meta"] = aex_meta.text;     out["aex.bin"] = std::move(aex.bytes);
    out["heads.meta"] = heads_meta.text; out["heads.bin"] = std::move(heads_bin.bytes);
    out["emb.bin"] = std::move(emb.bytes);
    out["stats_state_mean.bin"] = std::move(sm.bytes);
    out["stats_state_std.bin"] = std::move(ss.bytes);
    out["stats_action_mean.bin"] = std::move(am.bytes);
    out["stats_action_std.bin"] = std::move(as.bytes);
    out["config.txt"] = config_txt;
    return true;
}

} // namespace io
} // namespace tcpu
