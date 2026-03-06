// ane_convnext_bwd.h — ConvNeXt block backward pass + Adam weight update
//
// Forward: x → LN1 → dw → LN2 → pw1(C→4C) → SiLU → pw2(4C→C) → add(residual) → y
//
// Backward (chain rule, reverse order):
//   dy         = gradient w.r.t. block output
//   d_add_a    = dy                (add is just pass-through for both branches)
//   d_residual = dy                (residual branch: dx += dy)
//   d_pw2_y    = dy                (= d_add_a)
//   d_pw2      → dx2[4C,S], dW_pw2[C,4C]
//   d_gelu_y   = dx2
//   d_gelu     → d_gelu_x[4C,S]   (GELU backward, needs saved pre-GELU x)
//   d_pw1_y    = d_gelu_x
//   d_pw1      → dx1[C,S], dW_pw1[4C,C]
//   d_ln2_y    = dx1               (LN2 backward)
//   d_ln2      → d_dw_out[C,S]
//   d_dw       → dx_dw[C,S], dW_dw[C,K*K]   (NEON dw backward)
//   d_ln1_y    = dx_dw
//   d_ln1      → dx_block[C,S]    (gradient w.r.t. block input)
//   dx_total   = dx_block + d_residual   (add both paths)
//
// Weight updates via Adam after computing all gradients.
//
// Saved activations (needed for backward):
//   ln1_norm[C,S], ln1_rstd[S]   (from LN1)
//   dw_in[C,S]                   (= LN1 output, input to dw)
//   dw_out[C,S]                  (= dw output = LN2 input)
//   ln2_norm[C,S], ln2_rstd[S]   (from LN2)
//   pw1_x[C,S]                   (= LN2 output = pw1 input)
//   gelu_x[4C,S]                 (= pw1 output = GELU input)
//   pw2_x[4C,S]                  (= GELU output = pw2 input)
//
#pragma once
#include "ane_convnext.h"
#include "../ops/ane_matmul_bwd.h"
#include "../ops/ane_gelu_bwd.h"
#include "../ops/ane_silu_bwd.h"
#include "../ops/ane_fused_pw_bwd.h"
#include "../ops/ane_ln_bwd.h"
#include "../ops/ane_dw_bwd.h"
#include "../ops/ane_adam.h"
#include "../ops/ane_fused_silu_dw.h"
#include <stdlib.h>

typedef struct {
    int C, S, K;
    int checkpointed;  // 1 = recompute activations during bwd (saves memory, costs ~1 fwd pass)

    // Backward kernels
    ANEMatmulBwd *pw2_bwd;   // dy[C,S] → dx[4C,S], dW[C,4C]
    ANESiluBwd   *silu_bwd;
    ANEMatmulBwd *pw1_bwd;   // dy[4C,S] → dx[C,S], dW[4C,C]
    ANEFusedPwBwd *fused_dx; // fused pw2_dx+silu_bwd+pw1_dx (1 dispatch vs 3)
    ANEFusedSiluDw *fused_silu_dw; // fused silu+dW for pw2 (replaces pw2_dw when fwd uses fused_spa)

    // Optimizers (one per weight tensor)
    ANEAdam *opt_pw1;    // for W_pw1 [4C, C]
    ANEAdam *opt_pw2;    // for W_pw2 [C, 4C]
    ANEAdam *opt_dw;     // for W_dw  [C, K*K]  (CPU Adam — small)

    // Saved activations (NULL when checkpointed=1 — allocated on demand)
    _Float16 *ln1_norm;    // [C, S]
    _Float16 *ln1_rstd;    // [S]
    _Float16 *dw_in_saved; // [C, S]  (= ln1 output)
    _Float16 *dw_out_saved;// [C, S]  (= dw output, unused)
    _Float16 *ln2_norm;    // [C, S]
    _Float16 *ln2_rstd;    // [S]
    _Float16 *pw1_x;       // [C, S]  (= ln2 output)
    _Float16 *silu_x;      // [4C, S] (= pw1 output)
    _Float16 *pw2_x;       // [4C, S] (= gelu output)

    // Gradient scratch
    _Float16 *d_pw2_dx;   // [4C, S]
    _Float16 *d_silu_dx;  // [4C, S]
    _Float16 *d_pw1_dx;   // [C, S]
    _Float16 *d_ln2;      // [C, S]
    _Float16 *d_dw_dx;    // [C, S]
    _Float16 *d_ln1;      // [C, S]
    _Float16 *d_dw_dw;    // [C, K*K]  dW for depthwise
} ANEConvNeXtBwd;

static ANEConvNeXtBwd *ane_convnext_bwd_compile_ex(ANEConvNeXt *fwd,
                                                     float lr, float beta1, float beta2, float eps, float wd,
                                                     int checkpointed, IOSurfaceRef shared_lr_surf);

static ANEConvNeXtBwd *ane_convnext_bwd_compile(ANEConvNeXt *fwd,
                                                  float lr, float beta1, float beta2, float eps, float wd) {
    return ane_convnext_bwd_compile_ex(fwd, lr, beta1, beta2, eps, wd, 0, NULL);
}

static ANEConvNeXtBwd *ane_convnext_bwd_compile_ex(ANEConvNeXt *fwd,
                                                     float lr, float beta1, float beta2, float eps, float wd,
                                                     int checkpointed, IOSurfaceRef shared_lr_surf) {
    int C = fwd->C, S = fwd->S, K = fwd->K;
    ANEConvNeXtBwd *bwd = (ANEConvNeXtBwd *)calloc(1, sizeof(ANEConvNeXtBwd));
    bwd->C = C; bwd->S = S; bwd->K = K;
    bwd->checkpointed = checkpointed;

    // pw2 backward: Ci=4C, Co=C (still needed for dW kernel)
    bwd->pw2_bwd = ane_matmul_bwd_compile(C*4, C, S);
    // pw1 backward: Ci=C, Co=4C (still needed for dW kernel)
    bwd->pw1_bwd = ane_matmul_bwd_compile(C, C*4, S);
    // Standalone silu_bwd (fallback if fused fails)
    bwd->silu_bwd = ane_silu_bwd_compile(C*4, S);

    if (!bwd->pw2_bwd || !bwd->pw1_bwd || !bwd->silu_bwd) {
        fprintf(stderr, "ane_convnext_bwd_compile FAILED (C=%d S=%d)\n", C, S);
        free(bwd); return NULL;
    }

    // Try fused dx kernel: pw2_dx + silu_bwd + pw1_dx in 1 dispatch
    // pw1_out surface: fused_all populates its own pw1_out_surf, otherwise pw1->y_surf (= silu->x_surf)
    IOSurfaceRef pw1_out_for_bwd = fwd->fused_all ? fwd->fused_all->pw1_out_surf : fwd->silu->x_surf;
    bwd->fused_dx = ane_fused_pw_bwd_compile(C, C, S);
    if (bwd->fused_dx) {
        // Wire fused kernel inputs to forward weight + activation surfaces
        IOSurfaceRef fins[4] = {fwd->pw2->w_surf, fwd->pw1->w_surf, NULL, pw1_out_for_bwd};
        ane_rewire(bwd->fused_dx->k, fins, NULL);
        bwd->fused_dx->w_pw2_surf  = fwd->pw2->w_surf;
        bwd->fused_dx->w_pw1_surf  = fwd->pw1->w_surf;
        bwd->fused_dx->silu_x_surf = pw1_out_for_bwd;
        // Wire pw1_dw to use fused d_silu output as its dy input (compile-time, stable surface)
        {
            IOSurfaceRef ins_p[2] = {bwd->fused_dx->d_silu_surf, NULL};
            ane_rewire(bwd->pw1_bwd->k_dw, ins_p, NULL);
        }
    } else {
        fprintf(stderr, "  fused_pw_bwd FAILED — falling back to 3 separate dispatches\n");
        // Wire separate kernels as before
        ane_matmul_bwd_rewire_w(bwd->pw2_bwd, fwd->pw2->w_surf);
        ane_matmul_bwd_rewire_w(bwd->pw1_bwd, fwd->pw1->w_surf);
        // Chain: pw2_bwd.dx_surf → silu_bwd.dy_surf
        if (!bwd->pw2_bwd->cpu_mode) {
            IOSurfaceRef ins_g[2] = {bwd->pw2_bwd->dx_surf, NULL};
            ane_rewire(bwd->silu_bwd->k, ins_g, NULL);
            bwd->silu_bwd->dy_surf = bwd->pw2_bwd->dx_surf;
        }
        // Chain: silu_bwd.dx_surf → pw1_bwd.dy_surf
        if (!bwd->pw1_bwd->cpu_mode) {
            IOSurfaceRef ins_p[2] = {bwd->silu_bwd->dx_surf, NULL};
            ane_rewire(bwd->pw1_bwd->k_dx, ins_p, NULL);
            bwd->pw1_bwd->dy_surf = bwd->silu_bwd->dx_surf;
        }
        // Wire silu_bwd x_surf to fwd silu x_surf
        {
            IOSurfaceRef ins_x[2] = {bwd->silu_bwd->dy_surf, fwd->silu->x_surf};
            ane_rewire(bwd->silu_bwd->k, ins_x, NULL);
            bwd->silu_bwd->x_surf = fwd->silu->x_surf;
        }
    }

    // Wire bwd dW kernels to forward weight/activation surfaces (always needed)
    ane_matmul_bwd_rewire_w(bwd->pw2_bwd, fwd->pw2->w_surf);
    ane_matmul_bwd_rewire_w(bwd->pw1_bwd, fwd->pw1->w_surf);

    // When fwd uses fused_spa or fused_all, pw2_dw needs silu(pw1_out).
    // fwd->silu->y_surf won't be populated, so use fused_silu_dw instead.
    if (fwd->fused_spa || fwd->fused_all) {
        bwd->fused_silu_dw = ane_fused_silu_dw_compile(C, S);
        if (bwd->fused_silu_dw) {
            IOSurfaceRef sdins[2] = {NULL, pw1_out_for_bwd};
            ane_rewire(bwd->fused_silu_dw->k, sdins, NULL);
            bwd->fused_silu_dw->pw1_out_surf = pw1_out_for_bwd;
        }
    }

    // Optimizers — share lr_surf if provided (single write per step)
    bwd->opt_pw1 = ane_adam_compile_ex(C * C*4, lr, beta1, beta2, eps, wd, shared_lr_surf);
    bwd->opt_pw2 = ane_adam_compile_ex(C * C*4, lr, beta1, beta2, eps, wd, shared_lr_surf);
    // dw optimizer is CPU-side (small: C*K*K elements) — no lr_surf needed
    bwd->opt_dw  = ane_adam_compile(C * K*K, lr, beta1, beta2, eps, wd);

    if (!bwd->opt_pw1 || !bwd->opt_pw2 || !bwd->opt_dw) {
        fprintf(stderr, "ane_convnext_bwd_compile adam FAILED\n");
        free(bwd); return NULL;
    }

    // Wire optimizers to weight surfaces (pw1, pw2 share fwd weight surfaces)
    ane_adam_rewire_w(bwd->opt_pw1, fwd->pw1->w_surf);
    ane_adam_rewire_w(bwd->opt_pw2, fwd->pw2->w_surf);
    // Wire dW outputs from bwd kernels to optimizer dW inputs
    ane_adam_rewire_dw(bwd->opt_pw1, bwd->pw1_bwd->dw_surf);
    ane_adam_rewire_dw(bwd->opt_pw2,
        bwd->fused_silu_dw ? bwd->fused_silu_dw->dw_surf : bwd->pw2_bwd->dw_surf);
    // dw optimizer dW: CPU — written manually

    // Saved activation buffers — skipped in checkpointed mode (recomputed during bwd)
    // NOTE: silu_x is NOT saved — bwd->silu_bwd->x_surf is wired to fwd->silu->x_surf
    if (!checkpointed) {
        bwd->ln1_norm    = (_Float16 *)malloc((size_t)C * S * 2);
        bwd->ln1_rstd    = (_Float16 *)malloc((size_t)S * 2);
        bwd->dw_in_saved = (_Float16 *)malloc((size_t)C * S * 2);
        bwd->dw_out_saved= (_Float16 *)malloc((size_t)C * S * 2);
        bwd->ln2_norm    = (_Float16 *)malloc((size_t)C * S * 2);
        bwd->ln2_rstd    = (_Float16 *)malloc((size_t)S * 2);
        bwd->pw1_x       = (_Float16 *)malloc((size_t)C * S * 2);
        bwd->pw2_x       = (_Float16 *)malloc((size_t)C*4 * S * 2);
    } else {
        // Checkpointed: allocate only the buffers needed during recompute
        bwd->ln1_norm    = (_Float16 *)malloc((size_t)C * S * 2);
        bwd->ln1_rstd    = (_Float16 *)malloc((size_t)S * 2);
        bwd->dw_in_saved = (_Float16 *)malloc((size_t)C * S * 2);
        bwd->ln2_norm    = (_Float16 *)malloc((size_t)C * S * 2);
        bwd->ln2_rstd    = (_Float16 *)malloc((size_t)S * 2);
        // pw1_x, pw2_x, dw_out_saved: not needed — remain NULL
    }

    // Gradient scratch
    bwd->d_pw2_dx  = (_Float16 *)malloc((size_t)C*4 * S * 2);
    bwd->d_silu_dx = (_Float16 *)malloc((size_t)C*4 * S * 2);
    bwd->d_pw1_dx  = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->d_ln2     = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->d_dw_dx   = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->d_ln1     = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->d_dw_dw   = (_Float16 *)malloc((size_t)C * K*K * 2);

    return bwd;
}

// Call immediately after ane_convnext_eval() to save activations for backward.
static void ane_convnext_save_fwd(ANEConvNeXtBwd *bwd, ANEConvNeXt *fwd, int H) {
    // LN1 stats (small: [S] rstd + [C,S] norm)
    ane_ln_save_stats(fwd->ln1, bwd->ln1_norm, bwd->ln1_rstd);
    // dw_in = LN1 output — needed by dw backward
    memcpy(bwd->dw_in_saved, fwd->dw_in, (size_t)bwd->C * bwd->S * 2);
    // LN2 stats
    ane_ln_save_stats(fwd->ln2, bwd->ln2_norm, bwd->ln2_rstd);
    // silu_x: NOT saved — bwd->silu_bwd->x_surf wired to fwd->silu->x_surf (= pw1->y_surf)
    // pw1_x (ln2 out) and pw2_x (silu out) are rewired directly via IOSurface — no save needed.
    // dw_out_saved: not used in backward — skip.
}

// ---- Deferred dW/Adam work (overlaps with next block's dx computation) ----
#include <dispatch/dispatch.h>

static dispatch_queue_t _cnx_deferred_q = NULL;
static dispatch_group_t _cnx_deferred_g = NULL;

static void _cnx_deferred_init(void) {
    if (!_cnx_deferred_q) {
        _cnx_deferred_q = dispatch_queue_create("ane.cnx.deferred", DISPATCH_QUEUE_SERIAL);
        _cnx_deferred_g = dispatch_group_create();
    }
}

// Wait for all deferred dW/Adam work to complete.
// Must be called before the next forward pass (which reads updated weights).
static void ane_convnext_bwd_drain(void) {
    if (_cnx_deferred_g)
        dispatch_group_wait(_cnx_deferred_g, DISPATCH_TIME_FOREVER);
}

// Run backward pass. dy[C,S] is gradient from above. dx[C,S] receives gradient to pass below.
// t = current Adam step (1-indexed).
static double _cnx_now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e3+ts.tv_nsec*1e-6;}

static void ane_convnext_bwd_eval(ANEConvNeXtBwd *bwd, ANEConvNeXt *fwd,
                                   const _Float16 *dy, _Float16 *dx, int H, int t) {
    int C = bwd->C, S = bwd->S, K = bwd->K;
    static int _cnx_step = 0; ++_cnx_step;
    int _p = 0;  // set to (_cnx_step <= 28) for per-block timing
    double _t0, _t1;
#define _CT(label) if(_p){_t1=_cnx_now_ms();printf("    cnx %-20s %.1f ms\n",label,_t1-_t0);_t0=_t1;}
    if(_p) _t0=_cnx_now_ms();

    // 0. Checkpointed: recompute forward activations now (instead of reading from save_fwd)
    if (bwd->checkpointed) {
        // Re-run forward — populates ln1->out, ln1->k3c->out (rstd),
        // fwd->dw_in, ln2->out, ln2->k3c->out, gelu->x_surf, pw1/pw2 x_surf
        ane_convnext_eval(fwd, H);
        // Read the stats we need for backward
        ane_ln_save_stats(fwd->ln1, bwd->ln1_norm, bwd->ln1_rstd);
        memcpy(bwd->dw_in_saved, fwd->dw_in, (size_t)C * S * 2);
        ane_ln_save_stats(fwd->ln2, bwd->ln2_norm, bwd->ln2_rstd);
        // silu_x: not needed — bwd->silu_bwd->x_surf wired to fwd->silu->x_surf
    }
    if(_p){_CT("recompute_fwd");}

    // 1. Pointwise dx (critical path — produces d_pw1_dx for LN2 bwd)
    if (bwd->fused_dx) {
        ane_fused_pw_bwd_write_dy(bwd->fused_dx, dy);
        ane_fused_pw_bwd_eval(bwd->fused_dx);
        ane_fused_pw_bwd_read_dx(bwd->fused_dx, bwd->d_pw1_dx);
        if(_p){_CT("fused_pw_dx");}

        // 2. dW kernels: setup IO on main thread, defer ane_eval to overlap with NEON below
        if (bwd->fused_silu_dw) {
            IOSurfaceLock(bwd->fused_silu_dw->dy_surf, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(bwd->fused_silu_dw->dy_surf), dy, C*S*sizeof(_Float16));
            IOSurfaceUnlock(bwd->fused_silu_dw->dy_surf, 0, NULL);
        } else {
            ane_matmul_bwd_write_dy(bwd->pw2_bwd, dy);
            ane_matmul_bwd_rewire_x(bwd->pw2_bwd, fwd->pw2->x_surf);
        }
        ane_matmul_bwd_rewire_x(bwd->pw1_bwd, fwd->pw1->x_surf);
        // pw2_dw + pw1_dw eval deferred into the async block below
    } else {
        // Fallback: 3 separate dispatches for dx chain (not deferred — rare path)
        ane_matmul_bwd_write_dy(bwd->pw2_bwd, dy);
        ane_matmul_bwd_rewire_x(bwd->pw2_bwd, fwd->pw2->x_surf);
        ane_matmul_bwd_eval(bwd->pw2_bwd);
        if(_p){_CT("pw2_bwd");}
        if (bwd->pw2_bwd->cpu_mode) {
            ane_matmul_bwd_read_dx(bwd->pw2_bwd, bwd->d_pw2_dx);
            ane_silu_bwd_write_dy(bwd->silu_bwd, bwd->d_pw2_dx);
        }
        ane_silu_bwd_eval(bwd->silu_bwd);
        if(_p){_CT("silu_bwd");}
        if (bwd->pw1_bwd->cpu_mode) {
            ane_silu_bwd_read_dx(bwd->silu_bwd, bwd->d_silu_dx);
            ane_matmul_bwd_write_dy(bwd->pw1_bwd, bwd->d_silu_dx);
        }
        ane_matmul_bwd_rewire_x(bwd->pw1_bwd, fwd->pw1->x_surf);
        ane_matmul_bwd_eval(bwd->pw1_bwd);
        ane_matmul_bwd_read_dx(bwd->pw1_bwd, bwd->d_pw1_dx);
        if(_p){_CT("pw1_bwd");}
    }

    // 4. LN2 backward: d_pw1_dx[C,S] → d_dw_out[C,S]
    ane_ln_bwd(bwd->d_pw1_dx, bwd->ln2_norm, bwd->ln2_rstd, bwd->d_ln2, C, S);
    if(_p){_CT("ln2_bwd");}

    // 5. Depthwise backward: d_ln2[C,S] → d_ln1_out[C,S] + d_dw_dw[C,K*K]
    // d_ln2 is in [C,S] layout, dw operates in [C,H,H]
    _dw_neon_bwd_dx(bwd->d_ln2, fwd->dw_w, bwd->d_dw_dx, C, H, K);
    if(_p){_CT("dw_bwd_dx");}
    // dw_bwd_dw deferred — only needed by Adam, not on dx critical path

    // 6. LN1 backward: d_dw_dx[C,S] → dx_block[C,S]
    ane_ln_bwd(bwd->d_dw_dx, bwd->ln1_norm, bwd->ln1_rstd, bwd->d_ln1, C, S);
    if(_p){_CT("ln1_bwd");}

    // 7. Add residual branch: dx = d_ln1 + dy (NEON vectorized)
    {
        const _Float16 *a = bwd->d_ln1, *b = dy;
        int i = 0, n = C*S;
        for (; i + 7 < n; i += 8) {
            float16x8_t va = vld1q_f16((const __fp16*)(a+i));
            float16x8_t vb = vld1q_f16((const __fp16*)(b+i));
            vst1q_f16((__fp16*)(dx+i), vaddq_f16(va, vb));
        }
        for (; i < n; i++)
            dx[i] = (_Float16)((float)a[i] + (float)b[i]);
    }
    if(_p){_CT("residual_add");}
    // --- dx done. Next block can start. ---

    // 8. Deferred: dW computation + Adam + weight ping-pong.
    // These only update weights for the NEXT forward pass, so they can
    // run concurrently with the next block's dx computation.
    _cnx_deferred_init();
    // Wait for previous deferred block (serial queue ensures ordering)
    int _has_fused = (bwd->fused_dx != NULL);
    dispatch_group_async(_cnx_deferred_g, _cnx_deferred_q, ^{
        // pw2_dw + pw1_dw: ANE evals (IO already set up on main thread)
        if (_has_fused) {
            if (bwd->fused_silu_dw) {
                ane_fused_silu_dw_eval(bwd->fused_silu_dw);
            } else {
                ane_eval(bwd->pw2_bwd->k_dw);
            }
            ane_eval(bwd->pw1_bwd->k_dw);
        }

        // dw_bwd_dw: compute dW for depthwise (reads d_ln2 + dw_in_saved, both per-block)
        _dw_neon_bwd_dw(bwd->d_ln2, bwd->dw_in_saved, bwd->d_dw_dw, C, H, K);

        // pw1/pw2 Adam (CPU NEON for large weights)
        ane_adam_step(bwd->opt_pw1, t);
        ane_adam_step(bwd->opt_pw2, t);

        // dw: write dW + current W to Adam surfs, step, read back
        IOSurfaceLock(bwd->opt_dw->dw_surf, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(bwd->opt_dw->dw_surf), bwd->d_dw_dw,
               (size_t)C * K*K * 2);
        IOSurfaceUnlock(bwd->opt_dw->dw_surf, 0, NULL);
        IOSurfaceLock(bwd->opt_dw->w_surf, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(bwd->opt_dw->w_surf), fwd->dw_w, (size_t)C * K*K * 2);
        IOSurfaceUnlock(bwd->opt_dw->w_surf, 0, NULL);
        ane_adam_step(bwd->opt_dw, t);
        IOSurfaceLock(bwd->opt_dw->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        memcpy(fwd->dw_w, IOSurfaceGetBaseAddress(bwd->opt_dw->w_new_surf), (size_t)C * K*K * 2);
        IOSurfaceUnlock(bwd->opt_dw->w_new_surf, kIOSurfaceLockReadOnly, NULL);

        // pw1/pw2: ping-pong weight surfaces
#define _PINGPONG_W(opt, fwd_mm, bwd_mm) do { \
    if (!(opt)->cpu_mode) { \
        IOSurfaceRef _wu = (opt)->w_new_surf, _ws = (opt)->w_surf; \
        (opt)->w_surf = _wu; (opt)->w_new_surf = _ws; \
        (fwd_mm)->w_surf = _wu; \
        IOSurfaceRef _fi[2]={_wu,NULL}; ane_rewire((fwd_mm)->k, _fi, NULL); \
        IOSurfaceRef _ki[3]={_wu,NULL,NULL}, _ko[1]={_ws}; \
        ane_rewire((opt)->k_w, _ki, _ko); \
        ane_matmul_bwd_rewire_w(bwd_mm, _wu); \
    } else { \
        IOSurfaceLock((opt)->w_new_surf, kIOSurfaceLockReadOnly, NULL); \
        IOSurfaceLock((fwd_mm)->w_surf, 0, NULL); \
        memcpy(IOSurfaceGetBaseAddress((fwd_mm)->w_surf), \
               IOSurfaceGetBaseAddress((opt)->w_new_surf), \
               (size_t)(opt)->N * 2); \
        IOSurfaceUnlock((fwd_mm)->w_surf, 0, NULL); \
        IOSurfaceUnlock((opt)->w_new_surf, kIOSurfaceLockReadOnly, NULL); \
    } \
} while(0)
        _PINGPONG_W(bwd->opt_pw1, fwd->pw1, bwd->pw1_bwd);
        _PINGPONG_W(bwd->opt_pw2, fwd->pw2, bwd->pw2_bwd);
        if (bwd->fused_dx) {
            IOSurfaceRef fins[4] = {fwd->pw2->w_surf, fwd->pw1->w_surf, NULL, NULL};
            ane_rewire(bwd->fused_dx->k, fins, NULL);
            bwd->fused_dx->w_pw2_surf = fwd->pw2->w_surf;
            bwd->fused_dx->w_pw1_surf = fwd->pw1->w_surf;
        }
        if (fwd->fused_all) {
            IOSurfaceRef fins[4] = {fwd->pw1->w_surf, fwd->pw2->w_surf, NULL, NULL};
            ane_rewire(fwd->fused_all->k, fins, NULL);
            fwd->fused_all->w_pw1_surf = fwd->pw1->w_surf;
            fwd->fused_all->w_pw2_surf = fwd->pw2->w_surf;
        }
        if (fwd->fused_spa) {
            IOSurfaceRef spins[3] = {fwd->pw2->w_surf, NULL, NULL};
            ane_rewire(fwd->fused_spa->k, spins, NULL);
            fwd->fused_spa->w_surf = fwd->pw2->w_surf;
        }
#undef _PINGPONG_W
    });
#undef _CT
}
