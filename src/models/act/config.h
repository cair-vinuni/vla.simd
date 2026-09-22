/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// ACT (Action Chunking Transformer) dims, from tools/convert_act.py.
// See docs/10-act-design.md for the architecture these describe.

namespace tcpu {

struct ActConfig {
    int dim = 512, heads = 8, head_dim = 64, ff = 3200;
    int n_enc = 4, n_dec = 1;
    int chunk = 100, state_dim = 6, action_dim = 6;
    int n_1d = 2;             // learned 1D pos embeddings: latent + state token
    float ln_eps = 1e-5f;
};

} // namespace tcpu
