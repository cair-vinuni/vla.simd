/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "models/diffusion/diffusion_model.h"
#include "models/arena.h"
#include "hal/common/env.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tcpu {

static const float kNormEps = 1e-8f;

static void read_vec(std::ifstream& f, std::vector<float>& v, int n) {
    v.resize((size_t)n);
    f.read(reinterpret_cast<char*>(v.data()), sizeof(float)*(size_t)n);
}

bool DiffusionModel::load(const std::string& dir) {
    std::ifstream meta(dir + "/diffusion.meta");
    if (!meta) return false;

    std::string line;
    std::vector<int> dd;
    std::string sched_name = "DDPM";
    while (std::getline(meta, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if      (key == "n_obs_steps")      ss >> cfg.n_obs_steps;
        else if (key == "n_cams")           ss >> cfg.n_cams;
        else if (key == "img_h")            ss >> cfg.img_h;
        else if (key == "img_w")            ss >> cfg.img_w;
        else if (key == "crop_h")           ss >> cfg.crop_h;
        else if (key == "crop_w")           ss >> cfg.crop_w;
        else if (key == "resize_h")         ss >> cfg.resize_h;
        else if (key == "resize_w")         ss >> cfg.resize_w;
        else if (key == "state_dim")        ss >> cfg.state_dim;
        else if (key == "action_dim")       ss >> cfg.action_dim;
        else if (key == "horizon")          ss >> cfg.horizon;
        else if (key == "n_action_steps")   ss >> cfg.n_action_steps;
        else if (key == "num_keypoints")    ss >> cfg.num_keypoints;
        else if (key == "kernel_size")      ss >> cfg.kernel_size;
        else if (key == "n_groups")         ss >> cfg.n_groups;
        else if (key == "step_embed_dim")   ss >> cfg.step_embed_dim;
        else if (key == "num_train_timesteps") ss >> cfg.num_train_timesteps;
        else if (key == "num_inference_steps") ss >> cfg.num_inference_steps;
        else if (key == "beta_start")       ss >> cfg.beta_start;
        else if (key == "beta_end")         ss >> cfg.beta_end;
        else if (key == "beta_schedule")    ss >> cfg.beta_schedule;
        else if (key == "prediction_type")  ss >> cfg.prediction_type;
        else if (key == "clip_sample_range") ss >> cfg.clip_sample_range;
        else if (key == "gn_eps")           ss >> cfg.gn_eps;
        else if (key == "scheduler")        ss >> sched_name;
        else if (key == "film_scale")     { int v; ss >> v; cfg.film_scale = v != 0; }
        else if (key == "clip_sample")    { int v; ss >> v; cfg.clip_sample = v != 0; }
        else if (key == "separate_encoders") { int v; ss >> v; cfg.separate_encoder_per_camera = v != 0; }
        else if (key == "down_dims")      { int v; while (ss >> v) dd.push_back(v); }
        else if (key == "cam")            { std::string n; ss >> n; cam_names.push_back(n); }
    }
    if (!dd.empty()) cfg.down_dims = dd;
    cfg.scheduler = (sched_name == "DDIM") ? DPScheduler::DDIM : DPScheduler::DDPM;

    // The step count is a deployment knob and the env override is how the
    // ablation drives it. Read once per process: the engine caches nothing here,
    // but a caller that changes it mid-process would silently mix two ladders.
    const int steps = hal::env::int_env("DP_STEPS", 0);
    if (steps > 0) cfg.num_inference_steps = steps;
    if (const char* e = std::getenv("DP_SCHEDULER")) {
        cfg.scheduler = (std::string(e) == "DDIM") ? DPScheduler::DDIM : DPScheduler::DDPM;
    }

    if (cfg.n_cams <= 0 || cfg.n_obs_steps <= 0) return false;
    if (cfg.n_action_steps <= 0 || (long long)cfg.n_obs_steps - 1 + cfg.n_action_steps > cfg.horizon) return false;
    if (cfg.crop_h > cfg.img_h || cfg.crop_w > cfg.img_w) return false;
    if ((cfg.resize_h > 0 && cfg.resize_h != cfg.img_h) || (cfg.resize_w > 0 && cfg.resize_w != cfg.img_w))
        return false;
    if (cfg.num_inference_steps > cfg.num_train_timesteps) return false;

    const int n_enc = cfg.separate_encoder_per_camera ? cfg.n_cams : 1;
    encoders.resize((size_t)n_enc);
    for (int i=0; i<n_enc; i++) {
        std::string nm = "rgb_encoder" + std::to_string(i);
        if (!encoders[(size_t)i].load(dir, nm, cfg)) return false;
    }
    const int i8 = hal::env::int_env("DIFFUSION_INT8", 0);
    if (!unet.load(dir, "unet", cfg, (i8 & 1) != 0)) return false;
    if (!sched.init(cfg)) return false;
    if (i8) {
        int n = unet.n_int8;
        if (i8 & 2)
            for (DPRgbEncoder& e : encoders) n += e.backbone.quantize_convs(0, -1);
        std::fprintf(stderr, "[diffusion] int8 GEMMs: %d (DIFFUSION_INT8=%d)%s\n", n, i8,
                     n ? "" : " - no int8 kernel on this CPU, staying fp32");
    }

    // stats.bin: state min/max, action min/max, then per-camera image mean/std.
    // MIN_MAX for state and action is not a stylistic difference from the other
    // policies here -- it is what puts actions in [-1, 1] and makes the
    // scheduler's clip_sample_range of 1.0 the right number.
    std::ifstream st(dir + "/stats.bin", std::ios::binary);
    if (!st) return false;
    read_vec(st, state_min,  cfg.state_dim);
    read_vec(st, state_max,  cfg.state_dim);
    read_vec(st, action_min, cfg.action_dim);
    read_vec(st, action_max, cfg.action_dim);
    img_mean.resize((size_t)cfg.n_cams*3);
    img_std .resize((size_t)cfg.n_cams*3);
    for (int c=0; c<cfg.n_cams; c++) {
        st.read(reinterpret_cast<char*>(img_mean.data()+(size_t)c*3), sizeof(float)*3);
        st.read(reinterpret_cast<char*>(img_std .data()+(size_t)c*3), sizeof(float)*3);
    }
    if (!st) return false;

    // A converter that wrote identity statistics produces a model that looks
    // healthy in parity (both sides normalize the same way) and commands the
    // wrong thing on a robot. IMPACT shipped exactly that bug once; refuse it
    // here rather than discover it on hardware.
    bool degenerate = true;
    for (int i=0; i<cfg.action_dim; i++)
        if (action_max[(size_t)i] - action_min[(size_t)i] > 1e-6f) degenerate = false;
    if (degenerate) return false;

    return true;
}

void DiffusionModel::preprocess(const uint8_t* src, int cam, float* dst) const {
    // Normalize first, then crop -- the reference normalizes in the processor,
    // before the encoder's crop ever runs.
    const float* mean = img_mean.data()+(size_t)cam*3;
    const float* sd   = img_std .data()+(size_t)cam*3;

    const int in_h = cfg.img_h, in_w = cfg.img_w;
    const int out_h = cfg.crop_h > 0 ? cfg.crop_h : in_h;
    const int out_w = cfg.crop_w > 0 ? cfg.crop_w : in_w;

    // Center crop, matching torchvision CenterCrop at eval.
    const int dh = in_h - out_h, dw = in_w - out_w;
    const int top  = dh/2 + ((dh & 3) == 3);
    const int left = dw/2 + ((dw & 3) == 3);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int y=0; y<out_h; y++) {
        const uint8_t* srow = src + ((size_t)(y+top)*in_w + left)*3;
        float* drow = dst + (size_t)y*out_w*3;
        for (int x=0; x<out_w; x++)
            for (int c=0; c<3; c++)
                drow[x*3+c] = (srow[x*3+c]/255.0f - mean[c])/(sd[c] + kNormEps);
    }
}

void DiffusionModel::predict(const uint8_t* const* frames, const float* state,
                             const float* noise, bool unnormalize, float* actions) {
    const int S = cfg.n_obs_steps, N = cfg.n_cams;
    const int F = cfg.num_keypoints*2;
    const int H = cfg.horizon, A = cfg.action_dim;

    const int out_h = cfg.crop_h > 0 ? cfg.crop_h : cfg.img_h;
    const int out_w = cfg.crop_w > 0 ? cfg.crop_w : cfg.img_w;
    imgbuf.resize((size_t)out_h*out_w*3);

    // Global conditioning: per observation step, [state, cam0 feat, cam1 feat...],
    // concatenated over steps. The reference builds (B, S, state+cams*F) and
    // flattens, so the step axis is the slow one.
    const int per_step = cfg.state_dim + N*F;
    gcond.assign((size_t)per_step*S, 0.f);

    const size_t px = (size_t)cfg.img_h*cfg.img_w*3;
    enc_px.resize((size_t)S*N);
    enc_ft.resize((size_t)S*N);

    for (int s=0; s<S; s++) {
        float* g = gcond.data() + (size_t)s*per_step;
        for (int i=0; i<cfg.state_dim; i++) {
            const float den = state_max[(size_t)i] - state_min[(size_t)i];
            const float d   = den == 0.0f ? kNormEps : den;
            g[i] = 2.0f*(state[(size_t)s*cfg.state_dim + i] - state_min[(size_t)i])/d - 1.0f;
        }
        for (int c=0; c<N; c++) {
            const uint8_t* f = frames[(size_t)s*N + c];
            float* feat = g + cfg.state_dim + (size_t)c*F;
            const std::vector<float>* hit = nullptr;
            for (int k=0; k<S && !hit; k++) {
                const std::vector<uint8_t>& p = enc_px[(size_t)k*N + c];
                if (p.size() == px && !std::memcmp(p.data(), f, px)) hit = &enc_ft[(size_t)k*N + c];
            }
            if (hit) {
                std::memcpy(feat, hit->data(), sizeof(float)*(size_t)F);
            } else {
                preprocess(f, c, imgbuf.data());
                const DPRgbEncoder& enc =
                    encoders[cfg.separate_encoder_per_camera ? (size_t)c : 0];
                enc.forward(imgbuf.data(), scratch, feat);
            }
            std::vector<uint8_t>& slot_px = enc_px[(size_t)s*N + c];
            std::vector<float>& slot_ft = enc_ft[(size_t)s*N + c];
            if (hit != &slot_ft) {
                slot_px.clear();
                slot_ft.assign(feat, feat + F);
                slot_px.assign(f, f + px);
            }
        }
    }

    sample.assign((size_t)H*A, 0.f);
    if (noise) std::memcpy(sample.data(), noise, sizeof(float)*(size_t)H*A);
    eps.resize((size_t)H*A);

    for (size_t i=0; i<sched.timesteps.size(); i++) {
        const int t = sched.timesteps[i];
        unet.forward(eps.data(), sample.data(), gcond.data(), t);
        const float* step_noise = noise ? noise + (size_t)(i+1)*H*A : nullptr;
        sched.step(sample.data(), eps.data(), step_noise, t, (int)i);
    }

    // The robot executes n_action_steps starting at the current observation,
    // which sits at index n_obs_steps - 1 of the horizon -- not at 0. Slicing
    // from 0 would command the arm with actions for a frame it already passed.
    const int start = cfg.n_obs_steps - 1;
    for (int t=0; t<cfg.n_action_steps; t++) {
        const float* srcp = sample.data() + (size_t)(start + t)*A;
        float* dst = actions + (size_t)t*A;
        for (int i=0; i<A; i++) {
            if (unnormalize) {
                const float den = action_max[(size_t)i] - action_min[(size_t)i];
                const float d   = den == 0.0f ? kNormEps : den;
                dst[i] = (srcp[i] + 1.0f)/2.0f*d + action_min[(size_t)i];
            } else {
                dst[i] = srcp[i];
            }
        }
    }
}

} // namespace tcpu
