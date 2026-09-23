/*
 * Copyright 2026 FAI. Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * vla_simd.h - public surface of libvla_simd_{act,smolvla,octo,turbovla,impact,diffusion}.so.
 *
 * Ownership
 *   vla_<model>_load pairs with vla_<model>_free. Every other pointer - frames,
 *   state, tokens, the actions output - is caller-owned and borrowed only for
 *   the duration of the call. Nothing here allocates on the caller's behalf.
 *
 * Errors
 *   Fallible calls return vla_status (0 == VLA_OK); _load returns NULL. The
 *   exception is vla_smolvla_tokenize, which returns a token count >= 0. C++
 *   exceptions never cross the boundary: they are caught and reported as
 *   VLA_ERR_EXCEPTION, since unwinding through a ctypes caller's C frames
 *   terminates the process.
 *
 * Threading
 *   One handle is one inference at a time; the models carry mutable scratch.
 *   Concurrent calls on one handle need a lock (the reference servers hold one).
 *   Load a second handle for real concurrency.
 */

#ifndef VLA_SIMD_H
#define VLA_SIMD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define VLA_API __declspec(dllexport)
#else
#define VLA_API __attribute__((visibility("default")))
#endif

/* Bumped on any breaking change below; compare against vla_abi_version(). */
#define VLA_ABI_VERSION 1

typedef enum {
    VLA_OK            =  0,
    VLA_ERR_ARG       = -1,  /* NULL handle, NULL buffer, or an out-of-range dimension */
    VLA_ERR_SHAPE     = -2,  /* an argument disagrees with the loaded checkpoint */
    VLA_ERR_EXCEPTION = -3   /* a C++ exception was caught at the boundary */
} vla_status;

/* The library's VLA_ABI_VERSION, which may differ from the caller's header. */
VLA_API int32_t vla_abi_version(void);
VLA_API const char* vla_backend_name(void);
VLA_API int32_t vla_int8_available(void);

/* ---------------------------------------------------------------- ACT ----- */

VLA_API void* vla_act_load(const char* model_dir);
VLA_API void  vla_act_free(void* h);

/* 0 on a NULL handle. */
VLA_API int32_t vla_act_chunk(void* h);
VLA_API int32_t vla_act_action_dim(void* h);
VLA_API int32_t vla_act_state_dim(void* h);
VLA_API int32_t vla_act_n_cams(void* h);
VLA_API int32_t vla_act_img_h(void* h);
VLA_API int32_t vla_act_img_w(void* h);

/* images: n_cams frames back to back, uint8 HWC at img_h x img_w, in the
 * checkpoint's camera order. state [state_dim] in raw robot units.
 * actions out [chunk*action_dim]. */
VLA_API int32_t vla_act_predict(void* h, const uint8_t* images, const float* state,
                                int32_t unnormalize, float* actions);

/* ------------------------------------------------------------ SmolVLA ----- */

VLA_API void* vla_smolvla_load(const char* model_dir, const char* tok_dir);
VLA_API void  vla_smolvla_free(void* h);

VLA_API int32_t vla_smolvla_chunk(void* h);
VLA_API int32_t vla_smolvla_action_dim(void* h);
VLA_API int32_t vla_smolvla_state_dim(void* h);
VLA_API int32_t vla_smolvla_n_views(void* h);
VLA_API int32_t vla_smolvla_img_size(void* h);
VLA_API int32_t vla_smolvla_tok_maxlen(void* h);

/* text -> ids/mask, both [tok_maxlen], right-padded with pad_id / 0, as
 * lerobot's tokenizer_processor produces them. Returns the token count, or
 * VLA_ERR_ARG. Non-negative is success here, unlike the rest of the ABI. */
VLA_API int32_t vla_smolvla_tokenize(void* h, const char* text,
                                     int32_t* ids, int32_t* mask);

/* frames [n_views, height, width, 3] uint8 RGB at native resolution (the engine
 * does the resize-with-pad), in the order the checkpoint declares its image
 * features. state [state_dim] raw. noise [chunk*max_action_dim], or NULL to draw
 * from seed. actions out [chunk*action_dim], un-normalized. */
VLA_API int32_t vla_smolvla_predict(void* h, const uint8_t* frames, int32_t n_views,
                                    int32_t height, int32_t width,
                                    const int32_t* lang_tokens, const int32_t* lang_mask,
                                    int32_t n_lang, const float* state,
                                    const float* noise, uint64_t seed, float* actions);

/* --------------------------------------------------------------- Octo ----- */

VLA_API void* vla_octo_load(const char* model_dir, const char* tok_dir);
VLA_API void  vla_octo_free(void* h);

VLA_API int32_t vla_octo_horizon(void* h);
VLA_API int32_t vla_octo_action_dim(void* h);
VLA_API int32_t vla_octo_max_window(void* h);

/* primary [wnd,256,256,3] and wrist [wnd,128,128,3] uint8 HWC, pre-resized by
 * the caller. timestep_mask [wnd] (1 = real, 0 = history padding). wnd must be
 * in [1, vla_octo_max_window]. actions out [horizon*action_dim]. */
VLA_API int32_t vla_octo_predict(void* h, const uint8_t* primary, const uint8_t* wrist,
                                 int32_t wnd, const uint8_t* timestep_mask,
                                 const char* instruction, uint64_t seed,
                                 int32_t unnormalize, float* actions);

/* As vla_octo_predict, but with the DDPM noise supplied by the caller instead of
 * drawn from `seed`: noise [horizon*action_dim] is the initial x, z
 * [steps*horizon*action_dim] the per-step draw. Pass both or neither. Parity
 * against a stochastic reference is only meaningful with matched noise. */
VLA_API int32_t vla_octo_predict_ex(void* h, const uint8_t* primary, const uint8_t* wrist,
                                    int32_t wnd, const uint8_t* timestep_mask,
                                    const char* instruction, const float* noise,
                                    const float* z, uint64_t seed,
                                    int32_t unnormalize, float* actions);

/* ----------------------------------------------------------- TurboVLA ----- */

VLA_API void* vla_turbovla_load(const char* model_dir);
VLA_API void  vla_turbovla_free(void* h);

/* 0 on a NULL handle. */
VLA_API int32_t vla_turbovla_chunk(void* h);
VLA_API int32_t vla_turbovla_action_dim(void* h);
VLA_API int32_t vla_turbovla_state_dim(void* h);
VLA_API int32_t vla_turbovla_n_views(void* h);
VLA_API int32_t vla_turbovla_img_size(void* h);
VLA_API int32_t vla_turbovla_n_patches(void* h);   /* per view */
VLA_API int32_t vla_turbovla_text_pad(void* h);    /* padded text length */
VLA_API int32_t vla_turbovla_hidden(void* h);      /* fusion / decoder width */
VLA_API int32_t vla_turbovla_vis_dim(void* h);     /* DINOv3 width */
VLA_API int32_t vla_turbovla_text_hidden(void* h); /* BERT width */
VLA_API int32_t vla_turbovla_state_tokens(void* h);

/* The padded text length this instruction is pinned to (the checkpoint's
 * per-instruction table, else text_pad). VLA_ERR_ARG on a NULL argument. */
VLA_API int32_t vla_turbovla_pad_length(void* h, const char* instruction);

/* frames: n_views uint8 RGB HWC images at img_size x img_size, back to back, in
 * the checkpoint's view order (agentview, then wrist). They are consumed as
 * given - no resize and no rotation, which is what the reference policy
 * requires, so a LIBERO caller rotates its env frames first. state [state_dim]
 * in raw robot units. actions out [chunk*action_dim]; unnormalize maps them to
 * env units (arm through the dataset min/max, gripper to a hard +-1). */
VLA_API int32_t vla_turbovla_predict(void* h, const uint8_t* frames, const float* state,
                                     const char* instruction, int32_t unnormalize,
                                     float* actions);

/* Intermediates of the LAST vla_turbovla_predict on this handle, for module-level
 * parity against the reference dump. Integer tensors (token ids, masks) come
 * back as floats: every value is small enough to be exact in fp32. */
typedef enum {
    VLA_TURBOVLA_PIXEL_VALUES  =  0,  /* [n_views, 3, img, img]                */
    VLA_TURBOVLA_DINO_TOKENS   =  1,  /* [n_views, n_patches, vis_dim]         */
    VLA_TURBOVLA_VISION_PROJ   =  2,  /* [n_views, n_patches, hidden]          */
    VLA_TURBOVLA_VISUAL_TOKENS =  3,  /* [n_views*n_patches, hidden]           */
    VLA_TURBOVLA_INPUT_IDS     =  4,  /* [text_pad]                            */
    VLA_TURBOVLA_POSITION_IDS  =  5,  /* [text_pad]                            */
    VLA_TURBOVLA_TEXT_PAD_MASK =  6,  /* [text_pad], 1 = padding               */
    VLA_TURBOVLA_SELF_ATTN     =  7,  /* [text_pad, text_pad], 1 = attend      */
    VLA_TURBOVLA_BERT_HIDDEN   =  8,  /* [text_pad, text_hidden]               */
    VLA_TURBOVLA_TEXT_TOKENS   =  9,  /* [text_pad, hidden]                    */
    VLA_TURBOVLA_FUSED_VISUAL  = 10,  /* [n_views*n_patches, hidden]           */
    VLA_TURBOVLA_FUSED_TEXT    = 11,  /* [text_pad, hidden]                    */
    VLA_TURBOVLA_CONDITION     = 12,  /* [n_views*n_patches + text_pad, hidden]*/
    VLA_TURBOVLA_STATE_NORM    = 13,  /* [state_dim]                           */
    VLA_TURBOVLA_STATE_TOKENS  = 14,  /* [state_tokens, hidden]                */
    VLA_TURBOVLA_ACTIONS_NORM  = 15   /* [chunk, action_dim]                   */
} vla_turbovla_tensor_id;

/* Copies at most max_elems floats into out and returns how many were written,
 * VLA_ERR_ARG on a bad handle/pointer or unknown id, VLA_ERR_SHAPE when the
 * tensor does not fit (call with out = NULL and max_elems = 0 to size it: that
 * returns the element count). */
VLA_API int32_t vla_turbovla_tensor(void* h, int32_t which, float* out, int32_t max_elems);

/* ---------------------------------------------------------------------------
 * IMPACT - Instruction-Modulated Perception + ACTion chunking.
 *
 * ACT with a language tower: camera frames + joint state + an instruction ->
 * a chunk x action_dim action chunk, one forward pass.
 *
 * The instruction is passed on every predict() to match the other
 * language-conditioned engines, but it is only re-encoded when the string
 * changes - so a control loop repeating one sentence pays for the T5-small tower
 * and the FiLM head once per episode, not once per query.
 * --------------------------------------------------------------------------- */
VLA_API void* vla_impact_load(const char* model_dir);
VLA_API void  vla_impact_free(void* h);

VLA_API int32_t vla_impact_chunk(void* h);
VLA_API int32_t vla_impact_action_dim(void* h);
VLA_API int32_t vla_impact_state_dim(void* h);
VLA_API int32_t vla_impact_n_cams(void* h);
VLA_API int32_t vla_impact_img_h(void* h);
VLA_API int32_t vla_impact_img_w(void* h);
VLA_API int32_t vla_impact_n_text(void* h);   /* padded instruction length */

/* Encode an instruction and cache it for the episode. Optional: predict() does
 * this itself. Call it to keep the text tower out of a timed region. */
VLA_API int32_t vla_impact_set_instruction(void* h, const char* instruction);

/* frames [n_cams, img_h, img_w, 3] HWC RGB uint8 in the checkpoint's camera
 * order; state [state_dim] in raw robot units; actions [chunk, action_dim].
 * unnormalize != 0 applies the dataset stats. */
VLA_API int32_t vla_impact_predict(void* h, const uint8_t* frames, const float* state,
                                   const char* instruction, int32_t unnormalize,
                                   float* actions);

/* Language-side intermediates of the current instruction, for the parity harness.
 * ids/compact/mask are each [n_text] (any may be NULL); returns n_text. */
VLA_API int32_t vla_impact_tokens(void* h, int32_t* ids, int32_t* compact,
                                  int32_t* mask, int32_t max_elems);
/* gamma/beta are each [film_total] (either may be NULL); returns film_total. */
VLA_API int32_t vla_impact_film(void* h, float* gamma, float* beta, int32_t max_elems);

/* ---------------------------------------------------------------------------
 * Diffusion Policy (Chi et al. 2023), lerobot's implementation.
 *
 * The one model here that consumes an observation HISTORY rather than a single
 * frame: every query takes n_obs_steps frames per camera and n_obs_steps states.
 * That is why predict() takes no single-frame shortcut - a caller that passes
 * one frame is not running this policy, it is running a different one.
 *
 * The UNet runs once per denoising step, so latency scales with the step count.
 * DP_SCHEDULER ("DDPM"/"DDIM") and DP_STEPS override the checkpoint's own
 * choice, which is how the step ablation drives it.
 * --------------------------------------------------------------------------- */
VLA_API void* vla_diffusion_load(const char* model_dir);
VLA_API void  vla_diffusion_free(void* h);

VLA_API int32_t vla_diffusion_chunk(void* h);        /* n_action_steps, what the robot executes */
VLA_API int32_t vla_diffusion_horizon(void* h);      /* what the UNet denoises */
VLA_API int32_t vla_diffusion_action_dim(void* h);
VLA_API int32_t vla_diffusion_state_dim(void* h);
VLA_API int32_t vla_diffusion_n_cams(void* h);
VLA_API int32_t vla_diffusion_n_obs_steps(void* h);
VLA_API int32_t vla_diffusion_img_h(void* h);
VLA_API int32_t vla_diffusion_img_w(void* h);
VLA_API int32_t vla_diffusion_num_steps(void* h);    /* denoising steps actually run */
VLA_API int32_t vla_diffusion_is_ddim(void* h);

/* frames [n_obs_steps, n_cams, img_h, img_w, 3] HWC RGB uint8, oldest step
 * first; state [n_obs_steps, state_dim] raw units, oldest first;
 * noise [1 + num_steps, horizon, action_dim] - the prior followed by one buffer
 * per DDPM step (DDIM ignores all but the prior). Pass NULL only for shape
 * checks: with no prior the sampler starts from zeros, which is not a sample.
 * actions [chunk, action_dim]. */
VLA_API int32_t vla_diffusion_predict(void* h, const uint8_t* frames, const float* state,
                                      int32_t unnormalize, float* actions,
                                      const float* noise);

#ifdef __cplusplus
}
#endif

#endif /* VLA_SIMD_H */
