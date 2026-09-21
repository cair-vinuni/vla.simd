/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "turbovla_model.h"
#include "models/arena.h"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
using std::size_t;

namespace tcpu {
namespace {

// TURBOVLA_PROFILE=1: per-stage wall time of one predict() (stderr).
struct Profile {
    bool on = std::getenv("TURBOVLA_PROFILE") != nullptr;
    double t0 = 0;
    static double now_ms() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void tic() { if (on) t0 = now_ms(); }
    void toc(double& acc) { if (on) acc += now_ms() - t0; }
};

} // namespace

bool TurboVlaModel::load(const std::string& dir) {
    std::ifstream meta(dir + "/config.meta");
    if (!meta) { std::fprintf(stderr, "turbovla: cannot open %s/config.meta\n", dir.c_str()); return false; }
    std::string line;
    while (std::getline(meta, line)) {
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (key == "img_mean" || key == "img_std") {
            float* dst = key == "img_mean" ? cfg.img_mean : cfg.img_std;
            for (int i = 0; i < 3; i++) ls >> dst[i];
            continue;
        }
        float v = 0.0f;
        if (!(ls >> v)) continue;
        if      (key == "img"              ) cfg.img              = (int)v;
        else if (key == "n_views"          ) cfg.n_views          = (int)v;
        else if (key == "text_pad"         ) cfg.text_pad         = (int)v;
        else if (key == "max_text_len"     ) cfg.max_text_len     = (int)v;
        else if (key == "sub_sentence"     ) cfg.sub_sentence     = (int)v;
        else if (key == "chunk"            ) cfg.chunk            = (int)v;
        else if (key == "action_dim"       ) cfg.action_dim       = (int)v;
        else if (key == "state_dim"        ) cfg.state_dim        = (int)v;
        else if (key == "cls_id"           ) cfg.cls_id           = (int)v;
        else if (key == "sep_id"           ) cfg.sep_id           = (int)v;
        else if (key == "dot_id"           ) cfg.dot_id           = (int)v;
        else if (key == "question_id"      ) cfg.question_id      = (int)v;
        else if (key == "pad_id"           ) cfg.pad_id           = (int)v;
        else if (key == "unk_id"           ) cfg.unk_id           = (int)v;
        else if (key == "max_wordpiece"    ) cfg.max_wordpiece    = (int)v;
        else if (key == "gripper_deadband" ) cfg.gripper_deadband = v;
    }
    if (cfg.img < 1 || cfg.n_views < 1 || cfg.text_pad < 1 || cfg.text_pad > cfg.max_text_len ||
        cfg.chunk < 1 || cfg.action_dim < 1 || cfg.state_dim < 1) {
        std::fprintf(stderr, "turbovla: %s/config.meta shapes do not close (img %d views %d "
                     "text_pad %d/%d chunk %d action %d state %d)\n", dir.c_str(), cfg.img,
                     cfg.n_views, cfg.text_pad, cfg.max_text_len, cfg.chunk,
                     cfg.action_dim, cfg.state_dim);
        return false;
    }
    if (!cfg.sub_sentence) {
        // Every released checkpoint sets it. Without it BERT would take the plain
        // padding mask and implicit positions, which this loader does not build.
        std::fprintf(stderr, "turbovla: %s has sub_sentence 0, which this engine does "
                     "not implement\n", dir.c_str());
        return false;
    }

    if (!vision.load(dir, cfg.img)) return false;
    if (!fusion.load(dir)) return false;
    if (!text.load(dir, fusion.cfg.hidden)) return false;
    if (!head.load(dir)) return false;
    if (!tok.load(dir)) return false;

    // The four .meta files are written independently; a mismatched set would
    // otherwise show up as garbage actions rather than a load failure.
    if (fusion.cfg.vis_dim != vision.cfg.hidden || fusion.cfg.n_views != cfg.n_views ||
        head.cfg.hidden != fusion.cfg.hidden || head.cfg.chunk != cfg.chunk ||
        head.cfg.action_dim != cfg.action_dim || head.cfg.state_dim != cfg.state_dim) {
        std::fprintf(stderr, "turbovla: %s module dims disagree (vision %d vs fusion vis %d, "
                     "views %d vs %d, fusion hidden %d vs head %d)\n", dir.c_str(),
                     vision.cfg.hidden, fusion.cfg.vis_dim, cfg.n_views, fusion.cfg.n_views,
                     fusion.cfg.hidden, head.cfg.hidden);
        return false;
    }
    tok.cls_id = cfg.cls_id; tok.sep_id = cfg.sep_id;
    tok.pad_id = cfg.pad_id; tok.unk_id = cfg.unk_id;

    std::vector<float> stats;
    if (!read_arena(dir + "/stats.bin", stats)) {
        std::fprintf(stderr, "turbovla: cannot read %s/stats.bin\n", dir.c_str());
        return false;
    }
    const size_t S = (size_t)cfg.state_dim, A = (size_t)cfg.action_dim;
    if (stats.size() != 2*S + 2*A) {
        std::fprintf(stderr, "turbovla: %s/stats.bin has %zu floats, expected %zu\n",
                     dir.c_str(), stats.size(), 2*S + 2*A);
        return false;
    }
    proprio_mean.assign(stats.begin(),         stats.begin()+(std::ptrdiff_t)S);
    proprio_std .assign(stats.begin()+(std::ptrdiff_t)S,     stats.begin()+(std::ptrdiff_t)(2*S));
    action_min  .assign(stats.begin()+(std::ptrdiff_t)(2*S), stats.begin()+(std::ptrdiff_t)(2*S+A));
    action_max  .assign(stats.begin()+(std::ptrdiff_t)(2*S+A), stats.end());

    // text_pad.txt: "<length>\t<instruction>" per line. Optional - without it
    // every instruction uses config.meta's text_pad, which is what the reference
    // does for an instruction outside the table.
    std::ifstream pads(dir + "/text_pad.txt");
    if (pads) {
        while (std::getline(pads, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            const int len = std::atoi(line.substr(0, tab).c_str());
            if (len >= 1 && len <= cfg.max_text_len)
                pad_by_instruction[line.substr(tab+1)] = len;
        }
    }
    return true;
}

int TurboVlaModel::pad_length(const std::string& instruction) const {
    auto it = pad_by_instruction.find(instruction);
    return it == pad_by_instruction.end() ? cfg.text_pad : it->second;
}

// Tokenize, run BERT over the group length, zero-pad to text_pad, project.
// Everything the fusion stack needs about the text is left in the trace.
void TurboVlaModel::encode_text(const std::string& instruction) const {
    const int P = cfg.text_pad, TH = text.cfg.hidden, D = fusion.cfg.hidden;
    const int L = pad_length(instruction);          // <= P by construction

    int n_real = 0;
    const std::vector<int> ids = tok.encode(instruction, L, true, &n_real);
    ids_i.assign(ids.begin(), ids.end());
    pos_i.assign((size_t)L, 0);
    gmask.assign((size_t)L*L, 0.0f);
    text.build_masks(ids_i.data(), L, cfg, pos_i.data(), gmask.data());

    ghidden.assign((size_t)L*TH, 0.0f);
    text.encode(ids_i.data(), pos_i.data(), gmask.data(), L, ghidden.data());

    // Rows past the group length are literal zeros, which the projection turns
    // into its own bias - the reference builds the [P, hidden] block the same way.
    tr.bert_hidden.assign((size_t)P*TH, 0.0f);
    std::memcpy(tr.bert_hidden.data(), ghidden.data(), (size_t)L*TH*sizeof(float));
    tr.text_tokens.assign((size_t)P*D, 0.0f);
    text.project(tr.bert_hidden.data(), P, tr.text_tokens.data());

    tr.input_ids.assign((size_t)P, (float)cfg.pad_id);
    tr.position_ids.assign((size_t)P, 0.0f);
    tr.text_pad_mask.assign((size_t)P, 1.0f);
    tr.self_attn.assign((size_t)P*P, 0.0f);
    for (int i = 0; i < P; i++) tr.self_attn[(size_t)i*P + i] = 1.0f;   // identity outside the group
    for (int i = 0; i < L; i++) {
        tr.input_ids[(size_t)i] = (float)ids_i[(size_t)i];
        tr.position_ids[(size_t)i] = (float)pos_i[(size_t)i];
        tr.text_pad_mask[(size_t)i] = i < n_real ? 0.0f : 1.0f;
        for (int j = 0; j < L; j++)
            tr.self_attn[(size_t)i*P + j] = gmask[(size_t)i*L + j] == 0.0f ? 1.0f : 0.0f;
    }
}

void TurboVlaModel::predict(const uint8_t* frames, const float* state,
                            const std::string& instruction, bool unnormalize,
                            float* actions) const {
    const int V = cfg.n_views, IMG = cfg.img, NP = vision.n_patches();
    const int VD = vision.cfg.hidden, D = fusion.cfg.hidden, P = cfg.text_pad;
    const int NV = V*NP, C = head.cfg.chunk, A = head.cfg.action_dim;

    Profile pf;
    double t_pre = 0, t_vis = 0, t_txt = 0, t_proj = 0, t_fuse = 0, t_head = 0;
    const double t_start = Profile::now_ms();

    // uint8 HWC -> normalized CHW, the DINOv3 processor's math (rescale 1/255,
    // then per-channel mean/std). The reference refuses anything but a
    // pre-rotated img x img frame, so there is no resize here either.
    pf.tic();
    tr.pixel_values.resize((size_t)V*3*IMG*IMG);
    for (int w = 0; w < V; w++) {
        const uint8_t* src = frames + (size_t)w*IMG*IMG*3;
        float* dst = tr.pixel_values.data() + (size_t)w*3*IMG*IMG;
        for (int c = 0; c < 3; c++) {
            const float m = cfg.img_mean[c], s = cfg.img_std[c];
            for (int i = 0; i < IMG*IMG; i++)
                dst[(size_t)c*IMG*IMG + i] = (src[(size_t)i*3 + c]/255.0f - m)/s;
        }
    }
    pf.toc(t_pre);

    // encode_views_separately: the views share the tower's weights and never
    // attend each other, so they run as one batch with per-view attention.
    pf.tic();
    tr.dino_tokens.resize((size_t)NV*VD);
    vision.encode(tr.pixel_values.data(), V, tr.dino_tokens.data());
    pf.toc(t_vis);

    pf.tic();
    encode_text(instruction);
    pf.toc(t_txt);

    pf.tic();
    tr.vision_proj.resize((size_t)NV*D);
    tr.visual_tokens.resize((size_t)NV*D);
    fusion.project_vision(tr.dino_tokens.data(), V, NP, tr.visual_tokens.data(),
                          tr.vision_proj.data());
    pf.toc(t_proj);

    pf.tic();
    tr.fused_visual = tr.visual_tokens;
    tr.fused_text = tr.text_tokens;
    std::vector<uint8_t> pad((size_t)P);
    std::vector<float> tmask((size_t)P*P);
    for (int i = 0; i < P; i++) {
        pad[(size_t)i] = tr.text_pad_mask[(size_t)i] != 0.0f;
        for (int j = 0; j < P; j++)
            tmask[(size_t)i*P + j] = tr.self_attn[(size_t)i*P + j] != 0.0f
                                   ? 0.0f : -std::numeric_limits<float>::infinity();
    }
    fusion.forward(tr.fused_visual.data(), NV, tr.fused_text.data(), P,
                   pad.data(), tmask.data());

    tr.condition.resize((size_t)(NV + P)*D);
    std::memcpy(tr.condition.data(), tr.fused_visual.data(), (size_t)NV*D*sizeof(float));
    std::memcpy(tr.condition.data() + (size_t)NV*D, tr.fused_text.data(),
                (size_t)P*D*sizeof(float));
    pf.toc(t_fuse);

    pf.tic();
    tr.state_norm.resize((size_t)head.cfg.state_dim);
    for (int i = 0; i < head.cfg.state_dim; i++)
        tr.state_norm[(size_t)i] = (state[i] - proprio_mean[(size_t)i])
                                 / (proprio_std[(size_t)i] + 1e-6f);
    tr.state_tokens.resize((size_t)head.cfg.state_tokens*D);
    head.state_tokens(tr.state_norm.data(), tr.state_tokens.data());

    const int NM = NV + P + head.cfg.state_tokens;
    memory.resize((size_t)NM*D);
    std::memcpy(memory.data(), tr.condition.data(), (size_t)(NV+P)*D*sizeof(float));
    std::memcpy(memory.data() + (size_t)(NV+P)*D, tr.state_tokens.data(),
                (size_t)head.cfg.state_tokens*D*sizeof(float));

    tr.actions_norm.resize((size_t)C*A);
    head.decode(memory.data(), NM, tr.actions_norm.data());
    pf.toc(t_head);

    if (!unnormalize) {
        std::memcpy(actions, tr.actions_norm.data(), (size_t)C*A*sizeof(float));
    } else {
        // LIBERO eval protocol: the first 6 dims map from [-1,1] into the
        // dataset min/max; the gripper is a hard sign, with 0 counting as open.
        for (int t = 0; t < C; t++) {
            const float* in = tr.actions_norm.data() + (size_t)t*A;
            float* out = actions + (size_t)t*A;
            const int arm = A < 6 ? A : 6;
            for (int i = 0; i < arm; i++)
                out[i] = 0.5f*(in[i] + 1.0f)*(action_max[(size_t)i] - action_min[(size_t)i])
                       + action_min[(size_t)i];
            for (int i = arm; i < A; i++) out[i] = in[i];
            if (A > 6) out[6] = in[6] < -cfg.gripper_deadband ? -1.0f : 1.0f;
        }
    }

    if (pf.on) {
        const double wall = Profile::now_ms() - t_start;
        const double acc = t_pre + t_vis + t_txt + t_proj + t_fuse + t_head;
        std::fprintf(stderr,
            "  [turbovla] pre %5.1f  dinov3 x%d %7.1f  bert %6.1f  vproj %5.1f"
            "  fusion %6.1f  head %5.1f | other %5.1f | wall %7.1f ms\n",
            t_pre, V, t_vis, t_txt, t_proj, t_fuse, t_head, wall - acc, wall);
    }
}

} // namespace tcpu
