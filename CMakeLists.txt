cmake_minimum_required(VERSION 3.21...4.3)
project(vla_simd VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

# Every object ends up in one of the vla_simd_* shared libraries, so build
# the tree PIC and reuse the static libs instead of recompiling the HAL per .so.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Only the vla_* entry points are API; hiding the rest stops two engines in one
# process from interposing each other's C++ symbols.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Appended, not assigned: assigning CMAKE_CXX_FLAGS_RELEASE drops CMake's own
# `-O3 -DNDEBUG`, which left asserts live in a release build.
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -funroll-loops")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -funroll-loops")

# ---------------------------------------------------------------------------
# vla_flags: the ISA the HAL is compiled for. An INTERFACE target, not
# CMAKE_CXX_FLAGS_RELEASE, because src/hal/arch.h picks its backend from the
# macros these enable and Debug/RelWithDebInfo would otherwise fall through to
# the scalar kernels.
#
# x86-64 -> AVX2/FMA. Apple arm64 -> -mcpu. Other aarch64 -> baseline
# ARMv8 NEON. Anything else -> scalar fallback.
#
# Non-Apple aarch64 stays on the baseline by choice. Pi 5 (Cortex-A76, ACT, 4
# threads): armv8-a+simd 1031 ms, +dotprod+fp16 1043, -mtune=cortex-a76 1116,
# -mcpu=native 1120. The ISA features are ~free; the 8% is GCC's A76 cost model
# pessimizing the hand-written NEON kernels. Kernels wanting a newer extension
# (SDOT, AVX-VNNI) take it per-function via __attribute__((target(...))) behind
# a runtime check. -DVLA_ARM_NATIVE=ON opts in, -DVLA_ARM_FLAGS overrides.
# ---------------------------------------------------------------------------
add_library(vla_flags INTERFACE)
target_compile_features(vla_flags INTERFACE cxx_std_17)
target_include_directories(vla_flags INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  message(WARNING "vla_simd: no ISA flags for ${CMAKE_CXX_COMPILER_ID}; "
                  "the HAL will fall back to the scalar backend")
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)")
  # Apple is the one target that takes +dotprod from -mcpu rather than from the
  # pragma in src/hal/neon/gemm_i8.cpp, and native detects the real M-series
  # part. The virtualized CI runners resolve native to a CPU without dotprod, so
  # CI alone passes -DVLA_APPLE_MCPU=apple-m1.
  set(VLA_APPLE_MCPU "native" CACHE STRING "-mcpu for Apple arm64 (CI overrides with apple-m1)")
  target_compile_options(vla_flags INTERFACE -mcpu=${VLA_APPLE_MCPU})
  message(STATUS "vla_simd arm64: -mcpu=${VLA_APPLE_MCPU} (apple)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
  # Off by default: 8% slower than the baseline on a Cortex-A76 (see above).
  # The knob is here so another part can A/B it.
  option(VLA_ARM_NATIVE "Build aarch64 with -mcpu=native (build host only)" OFF)
  set(VLA_ARM_FLAGS "" CACHE STRING "override the aarch64 -march/-mcpu flags entirely")
  if(VLA_ARM_FLAGS)
    separate_arguments(VLA_ARM_FLAG_LIST NATIVE_COMMAND "${VLA_ARM_FLAGS}")
    target_compile_options(vla_flags INTERFACE ${VLA_ARM_FLAG_LIST})
    message(STATUS "vla_simd aarch64: ${VLA_ARM_FLAGS} (override)")
  elseif(VLA_ARM_NATIVE)
    target_compile_options(vla_flags INTERFACE -mcpu=native)
    message(STATUS "vla_simd aarch64: -mcpu=native")
  else()
    target_compile_options(vla_flags INTERFACE -march=armv8-a+simd)
    message(STATUS "vla_simd aarch64: baseline armv8-a+simd")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86")
  # AVX-VNNI is deliberately not global: it would let the compiler emit vpdpbusd
  # into any int8 loop and SIGILL pre-Alder-Lake / pre-Zen-4.
  # src/hal/avx2/gemm_i8.cpp opts in per-function behind a cpuid check.
  target_compile_options(vla_flags INTERFACE -mavx2 -mfma -mf16c)
  message(STATUS "vla_simd HAL: x86-avx2")
else()
  message(STATUS "vla_simd HAL: scalar (${CMAKE_SYSTEM_PROCESSOR})")
endif()

# GCC defaults to fast, Clang to on. Pin both to what the parity numbers were
# measured against.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(vla_flags INTERFACE -ffp-contract=fast)
endif()

# ---------------------------------------------------------------------------
# Sanitizers (dev only). -DVLA_SANITIZE=address,undefined
# ---------------------------------------------------------------------------
set(VLA_SANITIZE "" CACHE STRING "sanitizers to build with, e.g. address,undefined")
if(VLA_SANITIZE)
  target_compile_options(vla_flags INTERFACE
    -fsanitize=${VLA_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(vla_flags INTERFACE -fsanitize=${VLA_SANITIZE})
  message(STATUS "vla_simd sanitizers: ${VLA_SANITIZE}")
endif()

# Apple clang ships without OpenMP; point FindOpenMP at Homebrew's libomp.
if(APPLE AND NOT DEFINED OpenMP_CXX_FLAGS)
  execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
  endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(vla_flags INTERFACE OpenMP::OpenMP_CXX)
else()
  message(WARNING "vla_simd: OpenMP not found - the engine will run single-threaded "
                  "and every OMP_NUM_THREADS recipe in the docs is inert")
endif()

if(APPLE)
  # Accelerate sgemm (AMX) is the default Apple GEMM backend; TCPU_ACCEL=0 to
  # A/B it. NEW_LAPACK because the legacy CBLAS entry points are deprecated
  # since macOS 13.3.
  target_compile_definitions(vla_flags INTERFACE TCPU_ACCELERATE ACCELERATE_NEW_LAPACK=1)
  target_link_libraries(vla_flags INTERFACE "-framework Accelerate")
endif()

# ---------------------------------------------------------------------------
# HAL + shared layer (model-agnostic): reused by every VLA model
# ---------------------------------------------------------------------------
add_library(tcpu_ops STATIC
  src/hal/common/env.cpp
  src/hal/common/blas.cpp
  src/hal/common/gemm.cpp
  src/hal/common/gemm_i8.cpp
  src/hal/common/attn.cpp
  src/hal/avx2/gemm.cpp
  src/hal/avx2/gemm_i8.cpp
  src/hal/avx2/attn.cpp
  src/hal/apple/gemm.cpp
  src/hal/apple/attn.cpp
  src/hal/neon/gemm.cpp
  src/hal/neon/gemm_i8.cpp
  src/hal/neon/attn.cpp
  src/hal/scalar/gemm.cpp
  src/hal/scalar/attn.cpp
  src/hal/common/eltwise.cpp
  src/hal/common/norms.cpp
  src/hal/common/conv.cpp
  # nn modules over the HAL backends
  src/nn/linear.cpp
  src/nn/attention.cpp
  src/nn/encoder.cpp
  src/nn/t5_encoder.cpp
  src/nn/conv.cpp
  src/nn/diffusion.cpp
  src/nn/linear_bf16.cpp)   # load-time bf16 init, linked after the hot objects (see file header)
target_link_libraries(tcpu_ops PUBLIC vla_flags)

add_library(tcpu_preprocess STATIC src/preprocess/image.cpp)
target_link_libraries(tcpu_preprocess PUBLIC vla_flags)

add_library(tcpu_tokenizer STATIC src/tokenizer/tokenizer.cpp src/tokenizer/t5_tokenizer.cpp
                           src/tokenizer/bert_tokenizer.cpp)  # byte-level BPE (Llama-3 / SmolLM2) + sentencepiece unigram (T5) + BERT WordPiece
target_link_libraries(tcpu_tokenizer PUBLIC vla_flags)

# ---------------------------------------------------------------------------
# Models: one library per VLA architecture under src/models/<name>/
# ---------------------------------------------------------------------------
add_library(tcpu_octo STATIC
  src/models/octo/small_stem.cpp
  src/models/octo/octo_transformer.cpp
  src/models/octo/diffusion_head.cpp
  src/models/octo/octo_model.cpp)
target_link_libraries(tcpu_octo PUBLIC tcpu_ops tcpu_tokenizer)

add_library(tcpu_act STATIC
  src/models/act/resnet_backbone.cpp
  src/models/act/act_transformer.cpp
  src/models/act/act_model.cpp)
target_link_libraries(tcpu_act PUBLIC tcpu_ops)

add_library(tcpu_smolvla STATIC
  src/models/smolvla/smollm2_lm.cpp
  src/models/smolvla/siglip_vision.cpp
  src/models/smolvla/action_expert.cpp
  src/models/smolvla/smolvla_model.cpp)
target_link_libraries(tcpu_smolvla PUBLIC tcpu_ops tcpu_tokenizer tcpu_preprocess)

add_library(tcpu_turbovla STATIC
  src/models/turbovla/dinov3_vision.cpp
  src/models/turbovla/bert_text.cpp
  src/models/turbovla/fusion.cpp
  src/models/turbovla/action_head.cpp
  src/models/turbovla/turbovla_model.cpp)
target_link_libraries(tcpu_turbovla PUBLIC tcpu_ops tcpu_tokenizer)

add_library(tcpu_impact STATIC
  src/models/impact/t5_text.cpp
  src/models/impact/impact_model.cpp)
target_link_libraries(tcpu_impact PUBLIC tcpu_act tcpu_ops tcpu_tokenizer)

add_library(tcpu_diffusion STATIC
  src/models/diffusion/rgb_encoder.cpp
  src/models/diffusion/unet1d.cpp
  src/models/diffusion/scheduler.cpp
  src/models/diffusion/diffusion_model.cpp)
# The ResNet-18 image backbone IS ACT's, linked rather than copied - the
# reference builds both from the same torchvision resnet18 truncated at layer4.
target_link_libraries(tcpu_diffusion PUBLIC tcpu_act tcpu_ops)

# ---------------------------------------------------------------------------
# Shared libs with the C ABI (include/vla_simd.h) for ctypes/FFI callers. Each
# is its capi translation unit over the static libs, so there is no source list
# to keep in sync. No --whole-archive: every symbol is reachable from the
# exported entry points.
# ---------------------------------------------------------------------------
function(vla_add_shared name capi_src)
  add_library(${name} SHARED ${capi_src} src/capi_version.cpp)
  target_link_libraries(${name} PRIVATE ${ARGN})
  target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  add_library(vla::${name} ALIAS ${name})
  if(NOT SKBUILD)
    set_target_properties(${name} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${name} PRIVATE -fno-semantic-interposition)
  endif()
endfunction()

vla_add_shared(vla_simd_octo    src/models/octo/octo_capi.cpp       tcpu_octo)
vla_add_shared(vla_simd_act     src/models/act/act_capi.cpp         tcpu_act)
vla_add_shared(vla_simd_smolvla src/models/smolvla/smolvla_capi.cpp tcpu_smolvla)
vla_add_shared(vla_simd_turbovla src/models/turbovla/turbovla_capi.cpp tcpu_turbovla)
vla_add_shared(vla_simd_impact  src/models/impact/impact_capi.cpp    tcpu_impact)
vla_add_shared(vla_simd_diffusion src/models/diffusion/diffusion_capi.cpp tcpu_diffusion)

if(PROJECT_IS_TOP_LEVEL AND NOT SKBUILD)
  enable_testing()
  add_executable(test_ops tests/test_ops.cpp)
  target_link_libraries(test_ops PRIVATE tcpu_ops)
  add_test(NAME ops COMMAND test_ops)
  set_tests_properties(ops PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)
endif()

# ---------------------------------------------------------------------------
# Warnings, first-party targets only.
# ---------------------------------------------------------------------------
set(VLA_FIRST_PARTY
  tcpu_ops tcpu_preprocess tcpu_tokenizer tcpu_octo tcpu_act tcpu_smolvla tcpu_turbovla
  tcpu_impact tcpu_diffusion
  vla_simd_octo vla_simd_act vla_simd_smolvla vla_simd_turbovla vla_simd_impact
  vla_simd_diffusion)
if(TARGET test_ops)
  list(APPEND VLA_FIRST_PARTY test_ops)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  foreach(tgt IN LISTS VLA_FIRST_PARTY)
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wshadow)
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Link-time optimization. Off: measured neutral, and it slows debug and
# sanitizer links.
# ---------------------------------------------------------------------------
option(VLA_LTO "Link-time optimization in Release/RelWithDebInfo" OFF)
if(VLA_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT VLA_IPO_OK OUTPUT VLA_IPO_WHY)
endif()
if(VLA_LTO AND VLA_IPO_OK AND NOT VLA_SANITIZE)
  set_property(TARGET ${VLA_FIRST_PARTY}
               PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  set_property(TARGET ${VLA_FIRST_PARTY}
               PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
elseif(VLA_LTO)
  message(STATUS "vla_simd: LTO unavailable (${VLA_IPO_WHY})")
endif()


if(SKBUILD)
  install(TARGETS vla_simd_octo vla_simd_act vla_simd_smolvla vla_simd_turbovla
                  vla_simd_impact vla_simd_diffusion
          LIBRARY DESTINATION vla_simd)
  return()
endif()

# ---------------------------------------------------------------------------
# Install / export for find_package(vla_simd). Only the shared libs and the C
# header are API.
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS vla_simd_octo vla_simd_act vla_simd_smolvla vla_simd_turbovla
                vla_simd_impact vla_simd_diffusion
        EXPORT vla_simdTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/vla_simd.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vla_simdTargets NAMESPACE vla::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vla_simd)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/vla_simdConfig.cmake.in"
     "@PACKAGE_INIT@\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/vla_simdTargets.cmake\")\n")
configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/vla_simdConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vla_simdConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vla_simd)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/vla_simdConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vla_simdConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/vla_simdConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vla_simd)
