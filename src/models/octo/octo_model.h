/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn/t5_encoder.h"
#include "small_stem.h"
#include "octo_transformer.h"
#include "diffusion_head.h"
#include "tokenizer/t5_tokenizer.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Octo-Small orchestration: instruction + image window -> 4x7 action chunk.
// Weights from tools/octo/convert_octo.py, tokenizer from convert_t5_tokenizer.py.

namespace tcpu {

struct OctoModel {
    T5Tokenizer tok;
    nn::T5Encoder t5;
    SmallStem stem_primary, stem_wrist;
    OctoTransformer tf;
    DiffusionHead head;
    std::vector<float> act_mean, act_std, act_mask;   // [action_dim] dataset stats

    bool load(const std::string& dir, const std::string& tok_dir);

    // Quantize the matmul groups OCTO_INT8 selects to symmetric W8A8; called by
    // load(). Lossy and opt-in, exactly like ACT's and SmolVLA's - see the mask
    // comment in octo_model.cpp for the groups and what is deliberately left out.
    void apply_int8();

    // primary [wnd,256,256,3] uint8 HWC, wrist [wnd,128,128,3]; timestep_mask [wnd]
    // (0 = history padding). noise [flat], z [steps*flat]: pass nullptr to draw from
    // seed. actions [horizon, action_dim]; unnormalize applies dataset stats.
    // The T5 encoding is cached per instruction string (static factor, RAM-for-time).
    void predict(const uint8_t* primary, const uint8_t* wrist, int wnd,
                 const uint8_t* timestep_mask, const std::string& instruction,
                 const float* noise, const float* z, uint64_t seed,
                 bool unnormalize, float* actions) const;

    // Control-loop API: feed ONE new frame pair per step; the stem outputs of the
    // previous frame are reused from an internal window-2 history (frames are static
    // once encoded). First step runs as [repeat, real] with the pad mask, matching
    // octo's HistoryWrapper. Call reset_history() on a new episode.
    void predict_step(const uint8_t* primary, const uint8_t* wrist,
                      const std::string& instruction, uint64_t seed,
                      bool unnormalize, float* actions);

    // Split form of predict_step for pipelined loops: feed_frame() runs only the
    // stems (call it the moment a camera frame lands, e.g. from a capture thread -
    // it may overlap a running predict_fed), predict_fed() runs transformer + head
    // on the frames fed so far (no-op if none fed). One feeder thread at a time.
    // feed_frame + predict_fed back-to-back == predict_step.
    void feed_frame(const uint8_t* primary, const uint8_t* wrist);
    void predict_fed(const std::string& instruction, uint64_t seed,
                     bool unnormalize, float* actions);
    void reset_history() { hist_len = 0; }

private:
    const float* lang_encode(const std::string& instruction) const;
    void run_from_stems(const float* sp, const float* sw, int wnd,
                        const uint8_t* timestep_mask, const std::string& instruction,
                        const float* noise, const float* z, uint64_t seed,
                        bool unnormalize, float* actions) const;

    mutable std::unordered_map<std::string, std::vector<float>> lang_cache;
    std::mutex hist_mu;                    // guards hist_* against a feeder thread
    std::vector<float> hist_sp, hist_sw;   // stem outputs of the last 2 frames
    std::vector<float> feed_sp, feed_sw;   // stem scratch (outside the lock)
    std::vector<float> snap_sp, snap_sw;   // predict_fed window snapshot
    int hist_len = 0;
};

} // namespace tcpu
