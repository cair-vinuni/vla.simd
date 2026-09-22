/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "impact_model.h"
#include "hal/common/env.h"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
using std::size_t;

namespace tcpu {

bool ImpactModel::load(const std::string& dir) {
    std::ifstream cfgf(dir + "/config.txt");
    if (!cfgf) return false;
    int vocab_full = 32128, unk_id = 2;
    std::string line;
    cam_names.clear();
    while (std::getline(cfgf, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if      (key == "img_h"     ) ss >> img_h;
        else if (key == "img_w"     ) ss >> img_w;
        else if (key == "n_cams"    ) ss >> n_cams;
        else if (key == "norm_eps"  ) ss >> norm_eps;
        else if (key == "vocab_full") ss >> vocab_full;
        else if (key == "unk_id"    ) ss >> unk_id;
        else if (key.rfind("cam", 0) == 0) {
            std::string name;
            ss >> name;
            cam_names.push_back(name);
        }
    }

    if (!backbone.load(dir)) return false;
    if (!tf.load(dir)) return false;
    if (!tok.load(dir)) {
        std::fprintf(stderr, "impact: cannot load %s/vocab.txt\n", dir.c_str());
        return false;
    }
    if (!text.load(dir, vocab_full, unk_id)) return false;

    // The FiLM head and the backbone have to agree on how one flat gamma/beta
    // buffer is cut up. They are written by the same converter, so a mismatch
    // means the two arenas came from different runs - checked here rather than
    // discovered as a silently mis-modulated feature map.
    if (text.film_total() != backbone.film_total()) {
        std::fprintf(stderr, "impact: FiLM head emits %d channels but the backbone's "
                     "%zu points want %d - text.bin and vision.bin disagree\n",
                     text.film_total(), backbone.film_after.size(), backbone.film_total());
        return false;
    }
    if (text.n_text() != tf.cfg.n_text) {
        std::fprintf(stderr, "impact: text tower has n_text %d, transformer expects %d\n",
                     text.n_text(), tf.cfg.n_text);
        return false;
    }
    if (text.proj_dim() != tf.cfg.dim) {
        std::fprintf(stderr, "impact: text projection is %d wide, transformer is %d\n",
                     text.proj_dim(), tf.cfg.dim);
        return false;
    }

    std::ifstream st(dir + "/stats.bin", std::ios::binary);
    if (!st) return false;
    const int sd = tf.cfg.state_dim;
    const int ad = tf.cfg.action_dim;
    std::vector<float> raw((size_t)2*sd + 2*ad + (size_t)n_cams*6);
    st.read((char*)raw.data(), raw.size()*sizeof(float));
    if (!st) return false;

    const float* p = raw.data();
    state_mean.assign(p, p+sd);
    p += sd;
    state_std.assign(p, p+sd);
    p += sd;
    action_mean.assign(p, p+ad);
    p += ad;
    action_std.assign(p, p+ad);
    p += ad;
    img_mean.resize((size_t)n_cams*3);
    img_std.resize((size_t)n_cams*3);
    for (int c=0; c<n_cams; c++) {
        for (int i=0; i<3; i++) img_mean[(size_t)c*3+i] = *p++;
        for (int i=0; i<3; i++) img_std [(size_t)c*3+i] = *p++;
    }

    bscratch.resize(n_cams);
    norm.resize(n_cams);
    feats.resize(n_cams);

    apply_int8();
    return true;
}

// IMPACT_INT8 -> route matmul weights through the symmetric W8A8 kernel
// (ops/quant_ops.h). A bitmask, so a group can be A/B'd against fp32 on its own:
//
//   1  encoder attention projections (q/k/v/o)
//   2  encoder w1
//   4  encoder w2
//   8  token projections (image 1x1, state)
//   16 the whole decoder (self, cross, w1, w2)
//   32 ResNet convolutions, stem included
//   63 = all of it
//
// The groups mirror ACT_INT8's, because IMPACT is ACT plus a language tower and the
// interesting comparison is against ACT's ranking on the same device.
//
// **The text tower is deliberately not a group.** Its output is a pure function of
// the instruction and set_instruction() caches it, so in a control loop it runs once
// per episode and never again. Quantizing it would trade accuracy for a saving that
// amortizes to nothing - the same argument that keeps Octo's T5 out of OCTO_INT8.
// The FiLM head sits behind the same cache and is excluded for the same reason.
namespace {
enum : int { I8_ENC_ATTN = 1, I8_ENC_W1 = 2, I8_ENC_W2 = 4, I8_PROJ = 8,
             I8_DEC = 16, I8_CONV = 32 };

int impact_int8_mask() {
    static const int v = [] {
        const char* e = std::getenv("IMPACT_INT8");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}
} // namespace

// Quantize the groups IMPACT_INT8 selects. Runs after every sub-module has loaded,
// so one place decides and one line reports it. Layers whose shape the kernel cannot
// take (N % 16 != 0) stay fp32 silently, and on a CPU with no int8 kernel every call
// returns false - so a caller may set the mask unconditionally.
void ImpactModel::apply_int8() {
    const int sel = impact_int8_mask();
    if (!sel) return;

    int n = 0;
    for (ImpactEncoderLayer& l : tf.enc) {
        if (sel & I8_ENC_ATTN) {
            n += l.attn.wq.init_int8();
            n += l.attn.wk.init_int8();
            n += l.attn.wv.init_int8();
            n += l.attn.wo.init_int8();
        }
        if (sel & I8_ENC_W1) n += l.w1.init_int8();
        if (sel & I8_ENC_W2) n += l.w2.init_int8();
    }
    if (sel & I8_PROJ) {
        n += tf.img_proj.init_int8();
        n += tf.state_proj.init_int8();
    }
    if (sel & I8_DEC) {
        for (ImpactDecoderLayer& l : tf.dec) {
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
    if (sel & I8_CONV) {
        n += backbone.stem.init_int8();
        for (ImpactBasicBlock& b : backbone.blocks) {
            n += b.conv1.init_int8();
            n += b.conv2.init_int8();
            if (b.has_down) n += b.down.init_int8();
        }
    }
    std::fprintf(stderr, "[impact] int8 GEMMs: %d (IMPACT_INT8=%d)%s\n", n, sel,
                 n ? "" : " - no int8 kernel on this CPU, staying fp32");
}

bool ImpactModel::set_instruction(const std::string& instruction) {
    if (tf.cfg.dim < 1 || text.film_total() < 1) return false;
    if (have_text && instruction == cached_instruction) return true;
    const int L = tf.cfg.n_text;

    ids = tok.encode(instruction, L);
    ids.resize(L, tok.pad_id);

    // 1 for a real token, 0 for padding. The padded rows are still encoded - T5
    // produces a genuine hidden state for them - and are masked as keys later;
    // what this mask controls is where the tower may attend, which is the part
    // no downstream mask can repair.
    mask.assign(L, 0);
    n_real = 0;
    for (int i = 0; i < L; i++) {
        if (ids[(size_t)i] == tok.pad_id) break;
        mask[(size_t)i] = 1;
        n_real = i+1;
    }

    compact.assign(L, 0);
    text.remap(ids.data(), L, compact.data());

    text_tok.assign((size_t)L*tf.cfg.dim, 0.0f);
    gamma.assign((size_t)text.film_total(), 0.0f);
    beta.assign((size_t)text.film_total(), 0.0f);
    text_hidden.assign((size_t)L*text.d_model(), 0.0f);
    text.encode(compact.data(), mask.data(), L,
                text_tok.data(), gamma.data(), beta.data(), text_hidden.data());

    cached_instruction = instruction;
    have_text = true;
    return true;
}

void ImpactModel::encode_view(const uint8_t* image, int cam, float* feat) const {
    const size_t npx = (size_t)img_h*img_w;
    std::vector<float>& nx = norm[cam];
    if (nx.size() < npx*3) nx.resize(npx*3);

    // pixels -> [0,1] -> (x - mean)/(std + eps), same op order as the lerobot
    // NormalizerProcessorStep so the backbone sees bit-identical inputs
    const float* mean = img_mean.data()+(size_t)cam*3;
    const float* std_ = img_std.data()+(size_t)cam*3;
    float* dst = nx.data();
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t p=0; p<npx; p++)
        for (int c=0; c<3; c++)
            dst[p*3+c] = (image[p*3+c]/255.0f - mean[c])/(std_[c]+norm_eps);

    backbone.forward(nx.data(), img_h, img_w, bscratch[cam], feat,
                     gamma.empty() ? nullptr : gamma.data(),
                     beta.empty()  ? nullptr : beta.data());
}

void ImpactModel::predict(const uint8_t* const* images, const float* state,
                          bool unnormalize, float* actions) const {
    if (!have_text) {
        std::fprintf(stderr, "impact: predict() before set_instruction(). IMPACT is "
                     "instruction-conditioned; running it unconditioned would silently "
                     "produce a different policy.\n");
        return;
    }

    // IMPACT_PROFILE=1: per-stage latency on stderr
    static const bool prof = std::getenv("IMPACT_PROFILE") != nullptr;
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    };
    auto t0 = clk::now();

    int fh = 0, fw = 0;
    backbone.feat_size(img_h, img_w, &fh, &fw);
    const size_t nfeat = (size_t)fh*fw*backbone.out_channels();

    std::vector<const float*> fp(n_cams);
    for (int c=0; c<n_cams; c++) {
        if (feats[c].size() < nfeat) feats[c].resize(nfeat);
        fp[c] = feats[c].data();
    }

    // The cameras are independent. TCPU_VIEW_THREADS=N runs them concurrently on
    // N OMP threads each instead of one after another on the whole team; each
    // view already owns its own scratch, so the math is untouched either way.
    const int vt = hal::env::view_threads();
    if (vt > 0 && n_cams > 1) {
        std::vector<std::exception_ptr> err(n_cams);
        std::vector<std::thread> workers;
        workers.reserve(n_cams);
        try {
            for (int c=0; c<n_cams; c++) {
                workers.emplace_back([&, c] {
#if defined(_OPENMP)
                    omp_set_num_threads(vt);
#endif
                    try { encode_view(images[c], c, feats[c].data()); }
                    catch (...) { err[c] = std::current_exception(); }
                });
            }
        } catch (...) {
            for (std::thread& w : workers) w.join();
            throw;
        }
        for (std::thread& w : workers) w.join();
        for (std::exception_ptr& e : err) if (e) std::rethrow_exception(e);
    } else {
        for (int c=0; c<n_cams; c++)
            encode_view(images[c], c, feats[c].data());
    }
    auto t1 = clk::now();

    std::vector<float> sn(tf.cfg.state_dim);
    for (int i=0; i<tf.cfg.state_dim; i++)
        sn[i] = (state[i]-state_mean[i])/(state_std[i]+norm_eps);

    tf.forward(fp.data(), n_cams, fh, fw, sn.data(), text_tok.data(), text.text_pos,
               n_real, actions);
    auto t2 = clk::now();

    if (unnormalize) {
        for (int h=0; h<tf.cfg.chunk; h++)
            for (int a=0; a<tf.cfg.action_dim; a++) {
                float* v = actions+(size_t)h*tf.cfg.action_dim+a;
                *v = *v*action_std[a] + action_mean[a];
            }
    }

    if (prof)
        std::fprintf(stderr, "[impact] backbone %.1f ms (%d cams)  transformer %.1f ms  "
                             "total %.1f ms  (text cached: 0.0 ms)\n",
                     ms(t0, t1), n_cams, ms(t1, t2), ms(t0, t2));
}

} // namespace tcpu
