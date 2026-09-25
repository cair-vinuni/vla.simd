/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../arena.h"
#include "act_transformer.h"
#include "nn/encoder.h"
#include "ops/lm_ops.h"
#include "io/files.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
using std::size_t;

namespace tcpu {

// ACT_INT8 -> quantize the transformer GEMMs to W8A8 (see load()). A bitmask,
// so a layer group can be A/B'd against fp32 on its own: 1 encoder attention
// projections, 2 encoder w1, 4 encoder w2, 8 the camera-token projection,
// 16 the whole decoder. 31 = all of it.
enum : int { I8_ATTN = 1, I8_W1 = 2, I8_W2 = 4, I8_IMGPROJ = 8, I8_DEC = 16 };

bool ActTransformer::load(const std::string& dir, int img_ch, int int8_mask, bool int8_state) {
    io::InFile meta(dir + "/" + tag + ".meta");
    if (!meta) return false;
    std::string line;
    while (std::getline(meta, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if      (key == "dim"       ) ss >> cfg.dim;
        else if (key == "heads"     ) ss >> cfg.heads;
        else if (key == "head_dim"  ) ss >> cfg.head_dim;
        else if (key == "ff"        ) ss >> cfg.ff;
        else if (key == "n_enc"     ) ss >> cfg.n_enc;
        else if (key == "n_dec"     ) ss >> cfg.n_dec;
        else if (key == "chunk"     ) ss >> cfg.chunk;
        else if (key == "state_dim" ) ss >> cfg.state_dim;
        else if (key == "action_dim") ss >> cfg.action_dim;
        else if (key == "n_1d"      ) ss >> cfg.n_1d;
        else if (key == "n_text"    ) ss >> cfg.n_text;
        else if (key == "ln_eps"    ) ss >> cfg.ln_eps;
    }

    if (!read_arena(dir + "/" + tag + ".bin", data)) return false;

    // Shapes come from act.meta, the buffer size from act.bin. Linear::init packs
    // its weights immediately, so a short bin has to be caught before the init -
    // the off == data.size() check at the end of the walk is far too late.
    ArenaCursor<float> take{data};
    const int d = cfg.dim;
    auto take_linear = [&](nn::Linear& lin, int N, int K, nn::Linear::Role role) {
        const float* w = take((size_t)N*K);
        const float* b = take(N);
        if (!take.ok) return;
        lin.init(w, b, N, K, role);
    };
    auto take_mha = [&](nn::MhaQKV& a) {
        a.set_shape(cfg.heads, cfg.head_dim);
        take_linear(a.wq, d, d, nn::Linear::Role::Gemm);
        take_linear(a.wk, d, d, nn::Linear::Role::Gemm);
        take_linear(a.wv, d, d, nn::Linear::Role::Gemm);
        take_linear(a.wo, d, d, nn::Linear::Role::Gemm);
    };

    take_linear(img_proj, d, img_ch, nn::Linear::Role::Gemm);
    take_linear(state_proj, d, cfg.state_dim, nn::Linear::Role::Generic);
    latent_tok = take(d);
    pos1d      = take((size_t)cfg.n_1d*d);

    enc.resize(cfg.n_enc);
    for (ActEncoderLayer& l : enc) {
        take_mha(l.attn);
        l.n1s = take(d);
        l.n1b = take(d);
        take_linear(l.w1, cfg.ff, d, nn::Linear::Role::Mlp);
        take_linear(l.w2, d, cfg.ff, nn::Linear::Role::Mlp);
        l.n2s = take(d);
        l.n2b = take(d);
    }

    dec.resize(cfg.n_dec);
    for (ActDecoderLayer& l : dec) {
        take_mha(l.self);
        l.n1s = take(d);
        l.n1b = take(d);
        take_mha(l.cross);
        l.n2s = take(d);
        l.n2b = take(d);
        take_linear(l.w1, cfg.ff, d, nn::Linear::Role::Mlp);
        take_linear(l.w2, d, cfg.ff, nn::Linear::Role::Mlp);
        l.n3s = take(d);
        l.n3b = take(d);
    }

    dec_pos = take((size_t)cfg.chunk*d);
    dec_ns  = take(d);
    dec_nb  = take(d);
    take_linear(head, cfg.action_dim, d, nn::Linear::Role::Generic);
    if (!take.done() || cfg.n_1d != 2 || cfg.heads*cfg.head_dim != d) return false;

    // ACT_INT8=1 routes the transformer's GEMMs through the W8A8 kernel
    // (ops/quant_ops.h): every d x d projection and both MLPs, in the encoder,
    // the decoder and the camera-token projection. Left out on purpose:
    //   state_proj / head - K=6 and N=6, far too small to pay for quantizing,
    //                       and the head is the last op before the robot's
    //                       joint commands, so it keeps full fp32.
    //   attention itself  - scores and A*V are activation x activation, a
    //                       different quantization problem from weights.
    // Any layer whose shape the kernel cannot take silently stays fp32.
    if (int8_mask & 31) {
        int n = 0;
        if (int8_mask & I8_IMGPROJ) {
            n += img_proj.init_int8();
            if (int8_state) n += state_proj.init_int8();
        }
        for (ActEncoderLayer& l : enc) {
            if (int8_mask & I8_ATTN) {
                n += l.attn.wq.init_int8();
                n += l.attn.wk.init_int8();
                n += l.attn.wv.init_int8();
                n += l.attn.wo.init_int8();
            }
            if (int8_mask & I8_W1) n += l.w1.init_int8();
            if (int8_mask & I8_W2) n += l.w2.init_int8();
        }
        if (int8_mask & I8_DEC) {
            for (ActDecoderLayer& l : dec) {
                n += l.self.wq.init_int8();
                n += l.self.wk.init_int8();
                n += l.self.wv.init_int8();
                n += l.self.wo.init_int8();
                n += l.cross.wq.init_int8();
                n += l.cross.wk.init_int8();
                n += l.cross.wv.init_int8();
                n += l.cross.wo.init_int8();
                n += l.w1.init_int8();
                n += l.w2.init_int8();
            }
        }
        if (prof >= 2 || n == 0)
            std::fprintf(stderr, "[%s] int8 transformer GEMMs: %d%s\n", tag, n,
                         n ? "" : " (no int8 kernel on this CPU - staying fp32)");
    }
    return true;
}

void ActTransformer::sinusoid_pos_2d(int fh, int fw, int dim, float* out) {
    const int half = dim/2;
    const float two_pi = 6.283185307179586f;
    const float eps    = 1e-6f;

    std::vector<float> inv_freq(half);
    for (int c=0; c<half; c++)
        inv_freq[c] = std::pow(10000.0f, (float)(2*(c/2))/(float)half);

    for (int i=0; i<fh; i++) {
        const float yr = (float)(i+1)/((float)fh+eps)*two_pi;
        for (int j=0; j<fw; j++) {
            const float xr = (float)(j+1)/((float)fw+eps)*two_pi;
            float* o = out+(size_t)(i*fw+j)*dim;

            for (int c=0; c<half; c++) {
                const float y = yr/inv_freq[c];
                const float x = xr/inv_freq[c];
                o[c]      = (c & 1) ? std::cos(y) : std::sin(y);
                o[half+c] = (c & 1) ? std::cos(x) : std::sin(x);
            }
        }
    }
}

void ActTransformer::build_tokens(const float* const* feats, int n_cams, int fh, int fw,
                                  const float* state_norm, float* out_tokens, float* out_pos,
                                  const float* text, const float* text_pos, int n_text_real) const {
    const int d = cfg.dim;
    const int n_img = fh*fw;

    std::memcpy(out_tokens, latent_tok, sizeof(float)*d);
    state_proj.forward(out_tokens+d, state_norm, 1);
    std::memcpy(out_pos, pos1d, sizeof(float)*(size_t)cfg.n_1d*d);

    if (cam_pos_fh != fh || cam_pos_fw != fw) {
        cam_pos.resize((size_t)n_img*d);
        sinusoid_pos_2d(fh, fw, d, cam_pos.data());
        cam_pos_fh = fh;
        cam_pos_fw = fw;
    }
    for (int c=0; c<n_cams; c++)
        std::memcpy(out_pos+((size_t)cfg.n_1d+(size_t)c*n_img)*d, cam_pos.data(),
                    sizeof(float)*(size_t)n_img*d);

    for (int c=0; c<n_cams; c++)
        img_proj.forward(out_tokens+((size_t)cfg.n_1d+(size_t)c*n_img)*d, feats[c], n_img);

    // The text tail, real tokens only. The tower still encodes all cfg.n_text slots
    // - it has to, because T5 is bidirectional and the padded positions shape the
    // real ones' hidden states - but the padded rows stop here and never enter the
    // sequence. The tokens are the tower's projection; the position is its learned
    // table and goes in the pos array, not into the token, so that - as everywhere
    // else in this block - it reaches queries and keys but never values.
    const int nt = clamp_text(n_text_real);
    if (nt > 0) {
        const size_t text_off = ((size_t)cfg.n_1d + (size_t)n_cams*n_img)*d;
        std::memcpy(out_tokens+text_off, text,     sizeof(float)*(size_t)nt*d);
        std::memcpy(out_pos   +text_off, text_pos, sizeof(float)*(size_t)nt*d);
    }
}

void ActTransformer::encode(float* x, const float* tok_pos, int T) const {
    const int d = cfg.dim;
    const size_t n = (size_t)T*d;
    if (xq.size()    < n                 ) xq.resize(n);
    if (ff.size()    < (size_t)T*cfg.ff  ) ff.resize((size_t)T*cfg.ff);
    if (resid.size() < n) resid.resize(n);   // used by any layer without a fused epilogue

    // ACT_PROFILE=3 splits one encoder layer into its five pieces. Kept behind a
    // constant so the default path keeps its straight-line loop.
    const bool p3 = prof >= 3;
    double t_attn = 0, t_ln = 0, t_w1 = 0, t_w2 = 0, t_pos = 0;
    double tp = p3 ? nn::now_ms() : 0.0;
    auto tick = [&](double& acc) {
        if (!p3) return;
        const double now = nn::now_ms();
        acc += now-tp;
        tp = now;
    };

    for (const ActEncoderLayer& l : enc) {
        for (size_t i=0; i<n; i++)
            xq[i] = x[i]+tok_pos[i];
        tick(t_pos);

        if (l.attn.wo.add_ok()) {
            l.attn.forward(x, xq.data(), xq.data(), x, T, T, scratch, true);
        } else {
            l.attn.forward(resid.data(), xq.data(), xq.data(), x, T, T, scratch);
            for (size_t i=0; i<n; i++)
                x[i] += resid[i];
        }
        tick(t_attn);
        layernorm(x, x, l.n1s, l.n1b, T, d, cfg.ln_eps);
        tick(t_ln);

        l.w1.forward(ff.data(), x, T);
        relu(ff.data(), T*cfg.ff);
        tick(t_w1);
        if (l.w2.add_ok()) {
            l.w2.forward_add(x, ff.data(), T);
        } else {
            l.w2.forward(resid.data(), ff.data(), T);
            for (size_t i=0; i<n; i++)
                x[i] += resid[i];
        }
        tick(t_w2);
        layernorm(x, x, l.n2s, l.n2b, T, d, cfg.ln_eps);
        tick(t_ln);
    }

    if (p3)
        std::fprintf(stderr, "[%s]   enc: +tok_pos %.1f  attn(qkvo+scores) %.1f  "
                             "w1+relu %.1f  w2+add %.1f  layernorm %.1f ms\n",
                     tag, t_pos, t_attn, t_w1, t_w2, t_ln);
}

void ActTransformer::decode(const float* enc_out, const float* tok_pos, int T,
                            float* actions_norm) const {
    const int d = cfg.dim;
    const int C = cfg.chunk;
    const size_t n = (size_t)C*d;
    if (xq.size()    < n                 ) xq.resize(n);
    if (ff.size()    < (size_t)C*cfg.ff  ) ff.resize((size_t)C*cfg.ff);
    if (kpos.size()  < (size_t)T*d       ) kpos.resize((size_t)T*d);
    if (resid.size() < n                 ) resid.resize(n);

    // keys carry the encoder positions; values do not (and the encoder output is
    // the same for every decoder layer, so this is built once)
    for (size_t i=0; i<(size_t)T*d; i++)
        kpos[i] = enc_out[i]+tok_pos[i];

    if (dec_x.size() < n) dec_x.resize(n);
    std::fill(dec_x.begin(), dec_x.begin()+n, 0.0f);   // queries start at zero
    std::vector<float>& x = dec_x;
    for (const ActDecoderLayer& l : dec) {
        for (size_t i=0; i<n; i++)
            xq[i] = x[i]+dec_pos[i];

        if (l.self.wo.add_ok()) {
            l.self.forward(x.data(), xq.data(), xq.data(), x.data(), C, C, scratch, true);
        } else {
            l.self.forward(resid.data(), xq.data(), xq.data(), x.data(), C, C, scratch);
            for (size_t i=0; i<n; i++)
                x[i] += resid[i];
        }
        layernorm(x.data(), x.data(), l.n1s, l.n1b, C, d, cfg.ln_eps);

        for (size_t i=0; i<n; i++)
            xq[i] = x[i]+dec_pos[i];

        if (l.cross.wo.add_ok()) {
            l.cross.forward(x.data(), xq.data(), kpos.data(), enc_out, C, T, scratch, true);
        } else {
            l.cross.forward(resid.data(), xq.data(), kpos.data(), enc_out, C, T, scratch);
            for (size_t i=0; i<n; i++)
                x[i] += resid[i];
        }
        layernorm(x.data(), x.data(), l.n2s, l.n2b, C, d, cfg.ln_eps);

        l.w1.forward(ff.data(), x.data(), C);
        relu(ff.data(), C*cfg.ff);
        if (l.w2.add_ok()) {
            l.w2.forward_add(x.data(), ff.data(), C);
        } else {
            l.w2.forward(resid.data(), ff.data(), C);
            for (size_t i=0; i<n; i++)
                x[i] += resid[i];
        }
        layernorm(x.data(), x.data(), l.n3s, l.n3b, C, d, cfg.ln_eps);
    }

    layernorm(x.data(), x.data(), dec_ns, dec_nb, C, d, cfg.ln_eps);
    head.forward(actions_norm, x.data(), C);
}

void ActTransformer::forward(const float* const* feats, int n_cams, int fh, int fw,
                             const float* state_norm, float* actions_norm,
                             const float* text, const float* text_pos, int n_text_real) const {
    const int T = n_tokens(n_cams, fh, fw, n_text_real);
    const size_t n = (size_t)T*cfg.dim;
    // grow-only, like the rest of the scratch: a control loop runs this every step
    if (tokens.size() < n) tokens.resize(n);
    if (pos.size()    < n) pos.resize(n);

    double t0 = nn::now_ms();
    auto lap = [&](const char* what) {
        if (prof < 2) return;
        const double now = nn::now_ms();
        std::fprintf(stderr, "[%s]   %-22s %6.2f ms\n", tag, what, now-t0);
        t0 = now;
    };

    build_tokens(feats, n_cams, fh, fw, state_norm, tokens.data(), pos.data(),
                 text, text_pos, n_text_real);
    lap("build_tokens");
    encode(tokens.data(), pos.data(), T);
    lap("encode");
    decode(tokens.data(), pos.data(), T, actions_norm);
    lap("decode");
}

} // namespace tcpu
