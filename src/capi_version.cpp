/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

// Linked into every vla_simd_* library so each exports the handshake symbol.

#include "vla_simd.h"

extern "C" int32_t vla_abi_version(void) { return VLA_ABI_VERSION; }
