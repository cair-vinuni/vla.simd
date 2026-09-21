/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fusion.h"
#include "models/arena.h"
#include "ops/lm_ops.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
using std::size_t;

namespace tcpu {

bool TurboFusion::load(const std::string& dir) {
    std::ifstream meta(dir + "/fusion.meta");
    if (!meta) { std::fprintf(stderr, "turbovla: cannot open %s/fusion.meta\n", dir.c_str()); return false; }
    std::string key; float val;
    while (meta >> key >> val) {
        if      (key == "hidden"      ) cfg.hidden       = (int)val;
        else if (key == "embed"       ) cfg.embed        = (int)val;
        else if (key == "n_layers"    ) cfg.n_layers     = (int)val;
        else if (key == "fusion_heads") cfg.fusion_heads = (int)val;
        else if (key == "text_heads"  ) cfg.text_heads   = (int)val;
        else if (key == "text_ff"     ) cfg.text_ff      = (int)val;
        else if (key == "vis_dim"     ) cfg.vis_dim      = (int)val;
        else if (key == "vis_mlp"     ) cfg.vis_mlp      = (int)val;
        else if (key == "n_views"     ) cfg.n_views      = (int)val;
        else if (key == "ln_eps"      ) cfg.ln_eps       = val;
    }

    const bool bad = cfg.hidden < 1 || cfg.embed < 1 || cfg.n_layers < 1 ||
                     cfg.fusion_heads < 1 || cfg.text_heads < 1 || cfg.text_ff < 1 ||
                     cfg.vis_dim < 1 || cfg.vis_mlp < 1 || cfg.n_views < 1 ||
                     cfg.embed % cfg.fusion_heads != 0 ||
                     cfg.hidden % cfg.text_heads != 0;
    if (bad) {
        std::fprintf(stderr, "turbovla: %s/fusion.meta shapes do not close (hidden %d embed %d "
                     "layers %d fheads %d theads %d ff %d vis %d vmlp %d views %d)\n",
                     dir.c_str(), cfg.hidden, cfg.embed, cfg.n_layers, cfg.fusion_heads,
                     cfg.text_heads, cfg.text_ff, cfg.vis_dim, cfg.vis_mlp, cfg.n_views);
        return false;
    }

    if (!read_arena(dir + "/fusion.bin", data)) {
        std::fprintf(stderr, "turbovla: cannot read %s/fusion.bin\n", dir.c_str());
        return false;
    }
    const size_t D = (size_t)cfg.hidden, E = (size_t)cfg.embed, V = (size_t)cfg.vis_dim;
    const size_t M = (size_t)cfg.vis_mlp, F = (size_t)cfg.text_ff;
    const size_t want = 2*V + (M*V + M) + (D*M + D) + D*V + 2*D    // vision projection
                      + (size_t)cfg.n_views*D                      // view embedding
                      + (size_t)cfg.n_layers*(4*D                  // layer_norm_v/_l
                                              + 4*(E*D + E)        // v/l/values_v/values_l
                                              + 2*(D*E + D)        // out_v, out_l
                                              + 2*D                // gamma_v, gamma_l
                                              + 4*(D*D + D)        // text self-attention
                                              + 2*D                // norm1
                                              + (F*D + F)          // linear1
                                              + (D*F + D)          // linear2
                                              + 2*D);              // norm2
    if (data.size() != want) {
        std::fprintf(stderr, "turbovla: %s/fusion.bin has %zu floats, expected %zu\n",
                     dir.c_str(), data.size(), want);
        return false;
    }

    size_t off = 0;
    auto take = [&](size_t n) { const float* p = data.data()+off; off += n; return p; };
    using Role = nn::Linear::Role;

    in_norm_w = take(V); in_norm_b = take(V);
    const float* w1 = take(M*V); const float* b1 = take(M);
    const float* w2 = take(D*M); const float* b2 = take(D);
    const float* ws = take(D*V);
    out_norm_w = take(D); out_norm_b = take(D);
    view_emb = take((size_t)cfg.n_views*D);
    vp1.init(w1, b1, cfg.vis_mlp, cfg.vis_dim, Role::Mlp);
    vp2.init(w2, b2, cfg.hidden, cfg.vis_mlp, Role::Mlp);
    skip.init(ws, nullptr, cfg.hidden, cfg.vis_dim, Role::Gemm);   // skip has no bias

    layers.resize((size_t)cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; L++) {
        FusionLayer& l = layers[(size_t)L];
        l.ln_v_w = take(D); l.ln_v_b = take(D);
        l.ln_l_w = take(D); l.ln_l_b = take(D);
        const float* qvw = take(E*D); const float* qvbias = take(E);
        const float* qlw = take(E*D); const float* qlbias = take(E);
        const float* vvw = take(E*D); const float* vvbias = take(E);
        const float* vlw = take(E*D); const float* vlbias = take(E);
        const float* ovw = take(D*E); const float* ovb = take(D);
        const float* olw = take(D*E); const float* olb = take(D);
        l.gamma_v = take(D); l.gamma_l = take(D);
        l.qv.init(qvw, qvbias, cfg.embed, cfg.hidden, Role::Gemm);
        l.ql.init(qlw, qlbias, cfg.embed, cfg.hidden, Role::Gemm);
        l.vv.init(vvw, vvbias, cfg.embed, cfg.hidden, Role::Gemm);
        l.vl.init(vlw, vlbias, cfg.embed, cfg.hidden, Role::Gemm);
        l.out_v.init(ovw, ovb, cfg.hidden, cfg.embed, Role::Gemm);
        l.out_l.init(olw, olb, cfg.hidden, cfg.embed, Role::Gemm);

        const float* tqw = take(D*D); const float* tqb = take(D);
        const float* tkw = take(D*D); const float* tkb = take(D);
        const float* tvw = take(D*D); const float* tvb = take(D);
        const float* tow = take(D*D); const float* tob = take(D);
        l.n1_w = take(D); l.n1_b = take(D);
        const float* f1w = take(F*D); const float* f1b = take(F);
        const float* f2w = take(D*F); const float* f2b = take(D);
        l.n2_w = take(D); l.n2_b = take(D);
        l.text_attn.wq.init(tqw, tqb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.text_attn.wk.init(tkw, tkb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.text_attn.wv.init(tvw, tvb, cfg.hidden, cfg.hidden, Role::Gemm);
        l.text_attn.wo.init(tow, tob, cfg.hidden, cfg.hidden, Role::Gemm);
        l.text_attn.set_shape(cfg.text_heads, cfg.text_head_dim());
        l.ff1.init(f1w, f1b, cfg.text_ff, cfg.hidden, Role::Mlp);
        l.ff2.init(f2w, f2b, cfg.hidden, cfg.text_ff, Role::Mlp);
    }
    return true;
}

void TurboFusion::project_vision(const float* dino, int n_views, int n_patches,
                                 float* visual, float* proj_out) const {
    const int V = cfg.vis_dim, D = cfg.hidden, M = cfg.vis_mlp;
    const int n = n_views*n_patches;
    if (nv.size() < (size_t)n*V) nv.resize((size_t)n*V);
    if (ff.size() < (size_t)n*M) ff.resize((size_t)n*M);

    layernorm(nv.data(), dino, in_norm_w, in_norm_b, n, V, cfg.ln_eps);
    vp1.forward(ff.data(), nv.data(), n);
    gelu_erf(ff.data(), n*M);
    vp2.forward(visual, ff.data(), n);
    if (skip.add_ok()) {
        skip.forward_add(visual, dino, n);
    } else {
        if (nl.size() < (size_t)n*D) nl.resize((size_t)n*D);
        skip.forward(nl.data(), dino, n);
        for (size_t i = 0; i < (size_t)n*D; i++) visual[i] += nl[i];
    }
    layernorm(visual, visual, out_norm_w, out_norm_b, n, D, cfg.ln_eps);
    if (proj_out) std::memcpy(proj_out, visual, (size_t)n*D*sizeof(float));

    // + view embedding, broadcast over that view's patches (position_embedding
    // "view": no per-patch position embedding in this checkpoint family).
    for (int w = 0; w < n_views; w++) {
        const float* ve = view_emb + (size_t)w*D;
        for (int p = 0; p < n_patches; p++) {
            float* row = visual + (size_t)(w*n_patches + p)*D;
            for (int i = 0; i < D; i++) row[i] += ve[i];
        }
    }
}

void TurboFusion::forward(float* visual, int n_vis, float* text, int n_text,
                          const uint8_t* text_pad, const float* text_mask) const {
    const int D = cfg.hidden, E = cfg.embed, F = cfg.text_ff;
    const int fh = cfg.fusion_heads, fhd = cfg.fusion_head_dim();
    const float fscale = 1.0f/std::sqrt((float)fhd);
    const size_t VD = (size_t)n_vis*D, LD = (size_t)n_text*D;

    if (nv.size()  < VD)               nv.resize(VD);
    if (nl.size()  < LD)               nl.resize(LD);
    if (qvb.size() < (size_t)n_vis*E)  qvb.resize((size_t)n_vis*E);
    if (qlb.size() < (size_t)n_text*E) qlb.resize((size_t)n_text*E);
    if (vvb.size() < (size_t)n_vis*E)  vvb.resize((size_t)n_vis*E);
    if (vlb.size() < (size_t)n_text*E) vlb.resize((size_t)n_text*E);
    if (av.size()  < (size_t)n_vis*E)  av.resize((size_t)n_vis*E);
    if (al.size()  < (size_t)n_text*E) al.resize((size_t)n_text*E);
    if (dv.size()  < VD)               dv.resize(VD);
    if (dl.size()  < LD)               dl.resize(LD);
    if (ff.size()  < (size_t)n_text*F) ff.resize((size_t)n_text*F);

    // Vision queries text with the padded text columns blocked. One row of the
    // mask would do, but the attention op takes [seq_q, seq_k]; at 512 x 21 that
    // is 43 KB, built once for the whole stack.
    if (kmask.size() < (size_t)n_vis*n_text) kmask.resize((size_t)n_vis*n_text);
    {
        const float blocked = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n_text; j++) {
            const float m = (text_pad && text_pad[j]) ? blocked : 0.0f;
            for (int i = 0; i < n_vis; i++) kmask[(size_t)i*n_text + j] = m;
        }
    }

    for (int L = 0; L < cfg.n_layers; L++) {
        const FusionLayer& l = layers[(size_t)L];

        layernorm(nv.data(), visual, l.ln_v_w, l.ln_v_b, n_vis, D, cfg.ln_eps);
        layernorm(nl.data(), text,   l.ln_l_w, l.ln_l_b, n_text, D, cfg.ln_eps);

        l.qv.forward(qvb.data(), nv.data(), n_vis);     // vision queries / text keys
        l.ql.forward(qlb.data(), nl.data(), n_text);    // text keys / vision queries
        l.vv.forward(vvb.data(), nv.data(), n_vis);
        l.vl.forward(vlb.data(), nl.data(), n_text);

        gqa_attention_masked(av.data(), qvb.data(), qlb.data(), vlb.data(),
                             n_vis, n_text, fh, fh, fhd, fscale, kmask.data());
        // The other direction is the same score matrix transposed, so it uses
        // the same two projections with query and key swapped. No mask: nothing
        // pads the visual stream.
        gqa_attention_dense(al.data(), qlb.data(), qvb.data(), vvb.data(),
                            n_text, n_vis, fh, fh, fhd, fscale);
        l.out_v.forward(dv.data(), av.data(), n_vis);
        l.out_l.forward(dl.data(), al.data(), n_text);

        // residual_style "normalized": the residual base is the normalized
        // tensor the attention saw, not the block input.
        for (int t = 0; t < n_vis; t++) {
            float* o = visual + (size_t)t*D;
            const float* nrm = nv.data() + (size_t)t*D;
            const float* d = dv.data() + (size_t)t*D;
            for (int i = 0; i < D; i++) o[i] = nrm[i] + l.gamma_v[i]*d[i];
        }
        for (int t = 0; t < n_text; t++) {
            float* o = text + (size_t)t*D;
            const float* nrm = nl.data() + (size_t)t*D;
            const float* d = dl.data() + (size_t)t*D;
            for (int i = 0; i < D; i++) o[i] = nrm[i] + l.gamma_l[i]*d[i];
        }

        // Text encoder layer: post-norm, ReLU, sub-sentence attention mask. The
        // reference passes a key-padding mask too, but its TransformerEncoderLayer
        // never forwards it to the attention - only src_mask reaches the kernel.
        l.text_attn.forward(dl.data(), text, n_text, text_mask, -1, sc, nullptr);
        for (size_t i = 0; i < LD; i++) dl[i] += text[i];
        layernorm(text, dl.data(), l.n1_w, l.n1_b, n_text, D, cfg.ln_eps);

        l.ff1.forward(ff.data(), text, n_text);
        relu(ff.data(), n_text*F);
        l.ff2.forward(dl.data(), ff.data(), n_text);
        for (size_t i = 0; i < LD; i++) dl[i] += text[i];
        layernorm(text, dl.data(), l.n2_w, l.n2_b, n_text, D, cfg.ln_eps);
    }
}

} // namespace tcpu
