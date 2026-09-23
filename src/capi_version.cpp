/*
 * Copyright 2026 Khanh D. Nguyen, Hoang M. Truong, An T. Le.
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Linked into every vla_simd_* library so each exports the handshake symbol.

#include "vla_simd.h"
#include "hal/arch.h"
#include "ops/quant_ops.h"

extern "C" int32_t vla_abi_version(void) { return VLA_ABI_VERSION; }
extern "C" const char* vla_backend_name(void) { return tcpu::hal::backend_name(); }
extern "C" int32_t vla_int8_available(void) { return tcpu::int8_gemm_available() ? 1 : 0; }
