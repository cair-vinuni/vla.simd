/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// TurboVLA: tools/convert_turbovla.py fed from vla.cpp's GGUF. The GGUF already
// has LayerScale folded into o_proj/down_proj exactly as the converter folds it,
// and keeps nn.MultiheadAttention's packed in_proj, which is split here.
//
// What the GGUF does not carry:
//   * DINOv3's final LayerNorm (vit.norm), when absent. vla.cpp skips it;
//     upstream TurboVLA applies it (transformers >= 4.56 returns the normed
//     last_hidden_state as hidden_states[-1]). The tower is then marked
//     final_norm 0, so this engine computes what vla.cpp computes, not what the
//     checkpoint was trained with.
//   * The normalization statistics and the BERT vocabulary, which must sit
//     beside the GGUF as stats.bin and vocab.txt (vla-simd-serve stages both).

#include "io/gguf_models.h"
#include "io/json.h"
#include <algorithm>
#include <cstdio>
#include <sstream>

namespace tcpu {
namespace io {

namespace {

int count(const Gguf& g, const std::string& prefix, const std::string& suffix) {
    int n = 0;
    while (g.tensor(prefix + std::to_string(n) + suffix)) n++;
    return n;
}

// nn.MultiheadAttention's packed [3D, D] in_proj -> q, k, v, each weight then bias
void split_qkv(Blob& b, const std::vector<float>& w, const std::vector<float>& bias, int64_t D) {
    for (int i = 0; i < 3; i++) {
        b.f32(w.data() + (size_t)(i * D * D), (size_t)(D * D));
        b.f32(bias.data() + (size_t)(i * D), (size_t)D);
    }
}

size_t utf8_chars(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

} // namespace

bool adapt_turbovla(const Gguf& g, const Sidecar& side, Files& out, std::string& err) {
    TensorReader t{g, err};
    const std::string kv = "turbovla.";

    std::string vocab_txt, stats_bin;
    if (!side.read("vocab.txt", vocab_txt)) {
        err = g.path() + ": a TurboVLA GGUF carries no tokenizer; put bert-base-uncased's "
              "vocab.txt in " + side.dir;
        return false;
    }
    if (!side.read("stats.bin", stats_bin)) {
        err = g.path() + ": a TurboVLA GGUF carries no normalization statistics; put stats.bin "
              "(proprio mean, proprio std, action min, action max as float32) in " + side.dir +
              " (vla-simd-serve builds it from H-EmbodVis/TurboVLA libero_all4_stats.json)";
        return false;
    }

    Json cfg;
    if (!Json::parse(g.str(kv + "config_json", "{}"), cfg)) {
        err = g.path() + ": turbovla.config_json is not valid JSON";
        return false;
    }
    const Json* text_cfg = cfg.at("text");

    // ---- DINOv3 ViT ----
    const int64_t H = t.u32(kv + "vit_dim"), NH = t.u32(kv + "vit_heads");
    const int64_t patch = t.u32(kv + "patch_size"), regs = t.u32(kv + "num_register_tokens");
    const int VL = count(g, "vit.blk.", ".attn_q.weight");
    const std::vector<int64_t> fc1 = t.shape("vit.blk.0.fc1.weight");
    if (!t.ok()) return false;
    const int64_t I = fc1[0], PD = 3 * patch * patch;
    if (VL < 1 || NH < 1 || H % NH) { err = g.path() + ": DINOv3 geometry does not close"; return false; }
    const bool final_norm = g.tensor("vit.norm.weight") != nullptr;
    if (!final_norm)
        std::fprintf(stderr, "vla-simd: %s: no DINOv3 final norm in this GGUF; running without it, "
                     "as vla.cpp does (upstream TurboVLA applies it)\n", g.path().c_str());

    Meta vision_meta;
    vision_meta.i("hidden", H).i("n_heads", NH).i("head_dim", H / NH).i("inter", I)
               .i("n_layers", VL).i("patch", patch).i("prefix", regs + 1)
               .f("rope_theta", t.num(kv + "rope_theta")).f("ln_eps", 1e-5).i("final_norm", final_norm);
    Blob vision;
    vision.f32(t.f32("vit.cls_token", {1, H}));
    vision.f32(t.f32("vit.register_tokens", {regs, H}));
    vision.f32(t.f32("vit.patch_embed.weight", {H, PD}));
    vision.f32(t.f32("vit.patch_embed.bias", {H}));
    for (int L = 0; L < VL; L++) {
        const std::string p = "vit.blk." + std::to_string(L) + ".";
        vision.f32(t.f32(p + "ln1.weight", {H}));
        vision.f32(t.f32(p + "ln1.bias", {H}));
        vision.f32(t.f32(p + "attn_q.weight", {H, H}));
        vision.f32(t.f32(p + "attn_q.bias", {H}));
        vision.f32(t.f32(p + "attn_k.weight", {H, H}));
        // DINOv3 has key_bias=false; vla.cpp writes zeros in its place
        for (float v : t.f32(p + "attn_k.bias", {H}))
            if (v != 0.0f && t.ok()) err = g.path() + ": " + p + "attn_k.bias is not zero";
        vision.f32(t.f32(p + "attn_v.weight", {H, H}));
        vision.f32(t.f32(p + "attn_v.bias", {H}));
        vision.f32(t.f32(p + "attn_o.weight", {H, H}));
        vision.f32(t.f32(p + "attn_o.bias", {H}));
        vision.f32(t.f32(p + "ln2.weight", {H}));
        vision.f32(t.f32(p + "ln2.bias", {H}));
        vision.f32(t.f32(p + "fc1.weight", {I, H}));
        vision.f32(t.f32(p + "fc1.bias", {I}));
        vision.f32(t.f32(p + "fc2.weight", {H, I}));
        vision.f32(t.f32(p + "fc2.bias", {H}));
    }
    if (final_norm) {
        vision.f32(t.f32("vit.norm.weight", {H}));
        vision.f32(t.f32("vit.norm.bias", {H}));
    }
    if (!t.ok()) return false;

    // ---- BERT + text projection ----
    const int64_t TH = t.u32(kv + "text_dim"), TNH = t.u32(kv + "text_heads");
    const int TL = count(g, "text.encoder.layer.", ".attention.self.query.weight");
    const std::vector<int64_t> we = t.shape("text.embed.word_embeddings");
    const std::vector<int64_t> pe = t.shape("text.embed.position_embeddings");
    const std::vector<int64_t> te = t.shape("text.embed.token_type_embeddings");
    const std::vector<int64_t> ti = t.shape("text.encoder.layer.0.intermediate.dense.weight");
    const int64_t D = t.u32(kv + "hidden");
    if (!t.ok()) return false;
    const int64_t TI = ti[0];

    Meta text_meta;
    text_meta.i("hidden", TH).i("n_heads", TNH).i("head_dim", TH / TNH).i("inter", TI)
             .i("n_layers", TL).i("vocab", we[0]).i("max_pos", pe[0]).i("type_vocab", te[0])
             .f("ln_eps", 1e-12);
    Blob text;
    text.f32(t.f32("text.embed.word_embeddings", {we[0], TH}));
    text.f32(t.f32("text.embed.position_embeddings", {pe[0], TH}));
    text.f32(t.f32("text.embed.token_type_embeddings", {te[0], TH}));
    text.f32(t.f32("text.embed.LayerNorm.weight", {TH}));
    text.f32(t.f32("text.embed.LayerNorm.bias", {TH}));
    for (int L = 0; L < TL; L++) {
        const std::string p = "text.encoder.layer." + std::to_string(L) + ".";
        for (const char* n : {"attention.self.query", "attention.self.key", "attention.self.value",
                              "attention.output.dense"}) {
            text.f32(t.f32(p + n + ".weight", {TH, TH}));
            text.f32(t.f32(p + n + ".bias", {TH}));
        }
        text.f32(t.f32(p + "attention.output.LayerNorm.weight", {TH}));
        text.f32(t.f32(p + "attention.output.LayerNorm.bias", {TH}));
        text.f32(t.f32(p + "intermediate.dense.weight", {TI, TH}));
        text.f32(t.f32(p + "intermediate.dense.bias", {TI}));
        text.f32(t.f32(p + "output.dense.weight", {TH, TI}));
        text.f32(t.f32(p + "output.dense.bias", {TH}));
        text.f32(t.f32(p + "output.LayerNorm.weight", {TH}));
        text.f32(t.f32(p + "output.LayerNorm.bias", {TH}));
    }
    text.f32(t.f32("text_proj.weight", {D, TH}));
    text.f32(t.f32("text_proj.bias", {D}));
    if (!t.ok()) return false;

    // ---- vision projection, view embeddings, interaction layers ----
    const int64_t n_views = t.u32(kv + "num_views");
    const int FL = count(g, "vl_fusion.", ".v_proj.weight");
    const int XL = count(g, "vl_text.", ".attn_qkv.weight");
    const std::vector<int64_t> m0 = t.shape("vit_proj.mlp.0.weight");
    const std::vector<int64_t> vp = t.shape("vl_fusion.0.v_proj.weight");
    const std::vector<int64_t> xf = t.shape("vl_text.0.fc1.weight");
    if (!t.ok()) return false;
    if (FL != XL || FL < 1) {
        err = g.path() + ": " + std::to_string(FL) + " fusion layers but " + std::to_string(XL) +
              " text layers";
        return false;
    }
    const int64_t M = m0[0], E = vp[0], TF = xf[0];

    Meta fusion_meta;
    fusion_meta.i("hidden", D).i("embed", E).i("n_layers", FL)
               .i("fusion_heads", t.u32(kv + "fusion_heads"))
               .i("text_heads", t.u32(kv + "text_enhancer_heads")).i("text_ff", TF)
               .i("vis_dim", H).i("vis_mlp", M).i("n_views", n_views).f("ln_eps", 1e-5);
    Blob fusion;
    fusion.f32(t.f32("vit_proj.input_norm.weight", {H}));
    fusion.f32(t.f32("vit_proj.input_norm.bias", {H}));
    fusion.f32(t.f32("vit_proj.mlp.0.weight", {M, H}));
    fusion.f32(t.f32("vit_proj.mlp.0.bias", {M}));
    fusion.f32(t.f32("vit_proj.mlp.3.weight", {D, M}));
    fusion.f32(t.f32("vit_proj.mlp.3.bias", {D}));
    fusion.f32(t.f32("vit_proj.skip.weight", {D, H}));
    fusion.f32(t.f32("vit_proj.output_norm.weight", {D}));
    fusion.f32(t.f32("vit_proj.output_norm.bias", {D}));
    fusion.f32(t.f32("view_emb", {1, n_views, D}));
    for (int L = 0; L < FL; L++) {
        const std::string p = "vl_fusion." + std::to_string(L) + ".";
        fusion.f32(t.f32(p + "norm_v.weight", {D}));
        fusion.f32(t.f32(p + "norm_v.bias", {D}));
        fusion.f32(t.f32(p + "norm_l.weight", {D}));
        fusion.f32(t.f32(p + "norm_l.bias", {D}));
        for (const char* n : {"v_proj", "l_proj", "values_v", "values_l"}) {
            fusion.f32(t.f32(p + n + ".weight", {E, D}));
            fusion.f32(t.f32(p + n + ".bias", {E}));
        }
        for (const char* n : {"out_v", "out_l"}) {
            fusion.f32(t.f32(p + n + ".weight", {D, E}));
            fusion.f32(t.f32(p + n + ".bias", {D}));
        }
        fusion.f32(t.f32(p + "gamma_v", {D}));
        fusion.f32(t.f32(p + "gamma_l", {D}));

        const std::string x = "vl_text." + std::to_string(L) + ".";
        split_qkv(fusion, t.f32(x + "attn_qkv.weight", {3 * D, D}), t.f32(x + "attn_qkv.bias", {3 * D}), D);
        fusion.f32(t.f32(x + "attn_o.weight", {D, D}));
        fusion.f32(t.f32(x + "attn_o.bias", {D}));
        fusion.f32(t.f32(x + "ln1.weight", {D}));
        fusion.f32(t.f32(x + "ln1.bias", {D}));
        fusion.f32(t.f32(x + "fc1.weight", {TF, D}));
        fusion.f32(t.f32(x + "fc1.bias", {TF}));
        fusion.f32(t.f32(x + "fc2.weight", {D, TF}));
        fusion.f32(t.f32(x + "fc2.bias", {D}));
        fusion.f32(t.f32(x + "ln2.weight", {D}));
        fusion.f32(t.f32(x + "ln2.bias", {D}));
    }
    if (!t.ok()) return false;

    // ---- state projection, ACT decoder, action MLP ----
    const int64_t S = t.u32(kv + "state_dim"), A = t.u32(kv + "action_dim");
    const int64_t chunk = t.u32(kv + "action_horizon"), NS = t.u32(kv + "num_state_tokens");
    const int AL = count(g, "act.dec.", ".self_qkv.weight");
    const int PL = count(g, "act.proj.", ".weight");
    const std::vector<int64_t> s1 = t.shape("state.proj.1.weight");
    const std::vector<int64_t> dff = t.shape("act.dec.0.fc1.weight");
    const std::vector<int64_t> p0 = t.shape("act.proj.0.weight");
    if (!t.ok()) return false;
    const int64_t SH = s1[0], FF = dff[0], MH = p0[0];

    Meta head_meta;
    head_meta.i("hidden", D).i("n_layers", AL).i("n_heads", t.u32(kv + "action_heads")).i("ff", FF)
             .i("chunk", chunk).i("action_dim", A).i("state_dim", S).i("state_tokens", NS)
             .i("state_hidden", SH).i("mlp_hidden", MH).i("mlp_layers", PL).f("ln_eps", 1e-5);
    Blob head;
    head.f32(t.f32("state.proj.0.weight", {S}));
    head.f32(t.f32("state.proj.0.bias", {S}));
    head.f32(t.f32("state.proj.1.weight", {SH, S}));
    head.f32(t.f32("state.proj.1.bias", {SH}));
    head.f32(t.f32("state.proj.4.weight", {NS * D, SH}));
    head.f32(t.f32("state.proj.4.bias", {NS * D}));
    head.f32(t.f32("state.proj.position", {1, NS, D}));
    head.f32(t.f32("state.proj.output_norm.weight", {D}));
    head.f32(t.f32("state.proj.output_norm.bias", {D}));
    head.f32(t.f32("act.q.weight", {chunk, D}));
    for (int L = 0; L < AL; L++) {
        const std::string p = "act.dec." + std::to_string(L) + ".";
        split_qkv(head, t.f32(p + "self_qkv.weight", {3 * D, D}), t.f32(p + "self_qkv.bias", {3 * D}), D);
        head.f32(t.f32(p + "self_out.weight", {D, D}));
        head.f32(t.f32(p + "self_out.bias", {D}));
        head.f32(t.f32(p + "ln1.weight", {D}));
        head.f32(t.f32(p + "ln1.bias", {D}));
        split_qkv(head, t.f32(p + "cross_qkv.weight", {3 * D, D}), t.f32(p + "cross_qkv.bias", {3 * D}), D);
        head.f32(t.f32(p + "cross_out.weight", {D, D}));
        head.f32(t.f32(p + "cross_out.bias", {D}));
        head.f32(t.f32(p + "ln2.weight", {D}));
        head.f32(t.f32(p + "ln2.bias", {D}));
        head.f32(t.f32(p + "fc1.weight", {FF, D}));
        head.f32(t.f32(p + "fc1.bias", {FF}));
        head.f32(t.f32(p + "fc2.weight", {D, FF}));
        head.f32(t.f32(p + "fc2.bias", {D}));
        head.f32(t.f32(p + "ln3.weight", {D}));
        head.f32(t.f32(p + "ln3.bias", {D}));
    }
    for (int L = 0; L < PL; L++) {
        const std::string p = "act.proj." + std::to_string(L) + ".";
        const std::vector<int64_t> ws = t.shape(p + "weight");
        if (!t.ok()) return false;
        head.f32(t.f32(p + "weight", {ws[0], ws[1]}));
        head.f32(t.f32(p + "bias", {ws[0]}));
    }
    if (!t.ok()) return false;

    // ---- scalars, instruction padding table ----
    int64_t unk_id = 100;
    size_t max_wordpiece = 0, n_vocab = 0;
    {
        std::istringstream vs(vocab_txt);
        std::string tok;
        for (size_t i = 0; std::getline(vs, tok); i++) {
            if (tok == "[UNK]") unk_id = (int64_t)i;
            max_wordpiece = std::max(max_wordpiece, utf8_chars(tok));
            n_vocab = i + 1;
        }
    }
    if ((int64_t)n_vocab != we[0]) {
        err = side.dir + "/vocab.txt has " + std::to_string(n_vocab) + " tokens, the GGUF's BERT " +
              std::to_string(we[0]);
        return false;
    }
    const Json* max_len = text_cfg ? text_cfg->at("max_length") : nullptr;
    const Json* sub = text_cfg ? text_cfg->at("sub_sentence_present") : nullptr;

    Meta config_meta;
    config_meta.i("img", t.u32(kv + "image_size")).i("n_views", n_views)
               .i("text_pad", t.u32(kv + "max_text_length"))
               .i("max_text_len", max_len && max_len->is_num() ? (long long)max_len->as_num() : 256)
               .i("sub_sentence", sub && sub->is_num() ? (long long)sub->as_num() : 1)
               .i("chunk", chunk).i("action_dim", A).i("state_dim", S)
               .s("img_mean", "0.485 0.456 0.406").s("img_std", "0.229 0.224 0.225")
               .i("cls_id", t.u32(kv + "cls_token_id")).i("sep_id", t.u32(kv + "sep_token_id"))
               .i("dot_id", t.u32(kv + "period_token_id"))
               .i("question_id", t.u32(kv + "question_token_id"))
               .i("pad_id", t.u32(kv + "pad_token_id")).i("unk_id", unk_id)
               .i("max_wordpiece", (long long)max_wordpiece).f("gripper_deadband", 0.0);
    if (!t.ok()) return false;

    std::string text_pad;
    if (const Json* by = text_cfg ? text_cfg->at("padding_length_by_instruction") : nullptr)
        for (const auto& [instruction, len] : by->obj)          // std::map: sorted, as Python sorted()
            if (len.is_num()) text_pad += std::to_string((long long)len.as_num()) + "\t" + instruction + "\n";

    out["vision.meta"] = vision_meta.text; out["vision.bin"] = std::move(vision.bytes);
    out["text.meta"] = text_meta.text;     out["text.bin"] = std::move(text.bytes);
    out["fusion.meta"] = fusion_meta.text; out["fusion.bin"] = std::move(fusion.bytes);
    out["head.meta"] = head_meta.text;     out["head.bin"] = std::move(head.bytes);
    out["config.meta"] = config_meta.text;
    out["text_pad.txt"] = text_pad;
    return true;
}

} // namespace io
} // namespace tcpu
