// ane_ln_bwd.h — LayerNorm backward (CPU NEON)
//
// Forward LN: y = (x - mean) / sqrt(var + eps),  mean/var over C dim per token
// Layout: x [C, S]  (C channels, S tokens)
//
// Backward (standard LN gradient, per token s):
//   dx[c,s] = (1/C) * rstd[s] * (C*dy[c,s] - sum_c(dy[c,s]) - norm[c,s]*sum_c(dy[c,s]*norm[c,s]))
//   where norm[c,s] = (x[c,s]-mean[s])*rstd[s]
//
// CPU NEON: ANE has no reduction op for per-token sums over C in our layout.
// At C=96..384 and S=1024..16384 this is fast enough in practice.
//
// We need saved activations from forward:
//   norm[C,S] = (x - mean) / std  (= ln->out before scale/shift, but our LN has no affine)
//   rstd[S]   = 1/sqrt(var+eps)
//
// Since our ANELayerNorm has no scale/shift, dy = dout directly.
//
// Usage:
//   _Float16 *norm_buf = malloc(C*S*2);  // saved during forward
//   _Float16 *rstd_buf = malloc(S*2);
//   // fill during forward: ane_ln_save_stats(ln, norm_buf, rstd_buf)
//   ane_ln_bwd(dy, norm_buf, rstd_buf, dx, C, S);
//
#pragma once
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Backward through LayerNorm. All buffers [C,S] layout.
// Cache-friendly: iterate c outermost — dy[c,:] and norm[c,:] are contiguous rows.
// Pass 1: accumulate sum1[S], sum2[S] row-by-row (sequential reads).
// Pass 2: write dx[c,:] row-by-row (sequential writes).
// Scratch buffers for LN backward — pre-allocated to avoid calloc/free per call.
// Sized for max S across all stages. Caller must call ane_ln_bwd_init() once.
static float *_ln_bwd_sum1 = NULL;
static float *_ln_bwd_sum2 = NULL;
static float *_ln_bwd_scale = NULL;
static int    _ln_bwd_max_S = 0;

static void ane_ln_bwd_init(int max_S) {
    if (max_S <= _ln_bwd_max_S) return;
    free(_ln_bwd_sum1); free(_ln_bwd_sum2); free(_ln_bwd_scale);
    _ln_bwd_sum1  = (float *)malloc((size_t)max_S * sizeof(float));
    _ln_bwd_sum2  = (float *)malloc((size_t)max_S * sizeof(float));
    _ln_bwd_scale = (float *)malloc((size_t)max_S * sizeof(float));
    _ln_bwd_max_S = max_S;
}

static void __attribute__((noinline))
ane_ln_bwd(const _Float16 *dy, const _Float16 *norm, const _Float16 *rstd,
           _Float16 *dx, int C, int S) {
    float inv_C = 1.0f / (float)C;
    float fC = (float)C;

    // Auto-init if not called explicitly
    if (S > _ln_bwd_max_S) ane_ln_bwd_init(S);
    float *sum1 = _ln_bwd_sum1;
    float *sum2 = _ln_bwd_sum2;
    float *scale = _ln_bwd_scale;
    memset(sum1, 0, (size_t)S * sizeof(float));
    memset(sum2, 0, (size_t)S * sizeof(float));

    // Pass 1: sum1[s] = sum_c dy[c,s], sum2[s] = sum_c dy[c,s]*norm[c,s]
    // Tiled over S to keep sum1/sum2 slices in L1 cache.
    // Tile = 4096 floats = 16KB per array × 2 = 32KB, plus dy/norm reads ~16KB = ~48KB < 128KB L1
    #define LN_BWD_TILE 4096
    for (int s0 = 0; s0 < S; s0 += LN_BWD_TILE) {
        int s1 = s0 + LN_BWD_TILE < S ? s0 + LN_BWD_TILE : S;
        for (int c = 0; c < C; c++) {
            const _Float16 *dy_c   = dy   + c*S + s0;
            const _Float16 *norm_c = norm + c*S + s0;
            float *s1p = sum1 + s0, *s2p = sum2 + s0;
            int len = s1 - s0, s = 0;
            for (; s + 7 < len; s += 8) {
                float16x8_t dh = vld1q_f16((const __fp16*)(dy_c+s));
                float16x8_t nh = vld1q_f16((const __fp16*)(norm_c+s));
                float32x4_t dl = vcvt_f32_f16(vget_low_f16(dh));
                float32x4_t dhi = vcvt_f32_f16(vget_high_f16(dh));
                float32x4_t nl = vcvt_f32_f16(vget_low_f16(nh));
                float32x4_t nhi = vcvt_f32_f16(vget_high_f16(nh));
                vst1q_f32(s1p+s,   vaddq_f32(vld1q_f32(s1p+s),   dl));
                vst1q_f32(s1p+s+4, vaddq_f32(vld1q_f32(s1p+s+4), dhi));
                vst1q_f32(s2p+s,   vfmaq_f32(vld1q_f32(s2p+s),   dl,  nl));
                vst1q_f32(s2p+s+4, vfmaq_f32(vld1q_f32(s2p+s+4), dhi, nhi));
            }
            for (; s < len; s++) {
                float d=(float)dy_c[s], n=(float)norm_c[s];
                s1p[s]+=d; s2p[s]+=d*n;
            }
        }
    }
    #undef LN_BWD_TILE

    // Pre-compute scale[s] = min(rstd[s], RSTD_MAX) / C
    // Cap rstd to avoid LN backward amplifying nearly-constant tokens (std≈0).
    {
        const float RSTD_MAX = 10.0f;  // clamp rstd: std < 0.1 → cap amplification at 10×
        float32x4_t vinv = vdupq_n_f32(inv_C);
        float32x4_t vmax = vdupq_n_f32(RSTD_MAX);
        int s = 0;
        for (; s + 7 < S; s += 8) {
            float16x8_t rh = vld1q_f16((const __fp16*)(rstd+s));
            float32x4_t lo = vminq_f32(vcvt_f32_f16(vget_low_f16(rh)),  vmax);
            float32x4_t hi = vminq_f32(vcvt_f32_f16(vget_high_f16(rh)), vmax);
            vst1q_f32(scale+s,   vmulq_f32(lo, vinv));
            vst1q_f32(scale+s+4, vmulq_f32(hi, vinv));
        }
        for (; s < S; s++) {
            float r = (float)rstd[s]; if (r > RSTD_MAX) r = RSTD_MAX;
            scale[s] = r * inv_C;
        }
    }

    // Pass 2: dx[c,s] = scale[s] * (C*dy - sum1 - norm*sum2)
    // Tiled over S to keep scale/sum1/sum2 slices in L1
    float32x4_t vC = vdupq_n_f32(fC);
    #define LN_BWD_TILE2 4096
    for (int s0 = 0; s0 < S; s0 += LN_BWD_TILE2) {
        int s1 = s0 + LN_BWD_TILE2 < S ? s0 + LN_BWD_TILE2 : S;
        for (int c = 0; c < C; c++) {
            const _Float16 *dy_c   = dy   + c*S + s0;
            const _Float16 *norm_c = norm + c*S + s0;
            _Float16       *dx_c   = dx   + c*S + s0;
            float *scp = scale + s0, *s1p = sum1 + s0, *s2p = sum2 + s0;
            int len = s1 - s0, s = 0;
            for (; s + 7 < len; s += 8) {
                float16x8_t dh = vld1q_f16((const __fp16*)(dy_c+s));
                float16x8_t nh = vld1q_f16((const __fp16*)(norm_c+s));
                float32x4_t dl  = vcvt_f32_f16(vget_low_f16(dh));
                float32x4_t dhi = vcvt_f32_f16(vget_high_f16(dh));
                float32x4_t nl  = vcvt_f32_f16(vget_low_f16(nh));
                float32x4_t nhi = vcvt_f32_f16(vget_high_f16(nh));
                float32x4_t v0 = vmulq_f32(vld1q_f32(scp+s),
                    vsubq_f32(vsubq_f32(vmulq_f32(vC, dl), vld1q_f32(s1p+s)),
                               vmulq_f32(nl, vld1q_f32(s2p+s))));
                float32x4_t v1 = vmulq_f32(vld1q_f32(scp+s+4),
                    vsubq_f32(vsubq_f32(vmulq_f32(vC, dhi), vld1q_f32(s1p+s+4)),
                               vmulq_f32(nhi, vld1q_f32(s2p+s+4))));
                vst1q_f16((__fp16*)(dx_c+s), vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
            }
            for (; s < len; s++) {
                float d=(float)dy_c[s], n=(float)norm_c[s];
                dx_c[s]=(_Float16)(scp[s]*(fC*d - s1p[s] - n*s2p[s]));
            }
        }
    }
    #undef LN_BWD_TILE2
}

// Read norm (= LN output = normalized x) and rstd from ANE surfaces after forward.
// norm_buf[C,S] and rstd_buf[S] must be pre-allocated by caller.
// Call immediately after ane_ln_eval() while surfaces still hold the values.
//
// ANELayerNorm internals:
//   k4->ioOutputs[0] = NORM [C,S]  (= ln->out)
//   k3c->ioOutputs[0] = RSTD [1,S] (= 1/sqrt(var+eps))
#include "ane_ln.h"

static void ane_ln_save_stats(ANELayerNorm *ln, _Float16 *norm_buf, _Float16 *rstd_buf) {
    // norm is ln->out (k4 output = NORM)
    IOSurfaceLock(ln->out, kIOSurfaceLockReadOnly, NULL);
    memcpy(norm_buf, IOSurfaceGetBaseAddress(ln->out), ln->C * ln->S * sizeof(_Float16));
    IOSurfaceUnlock(ln->out, kIOSurfaceLockReadOnly, NULL);
    // rstd is k3c output
    IOSurfaceLock(ln->k3c->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(rstd_buf, IOSurfaceGetBaseAddress(ln->k3c->ioOutputs[0]), ln->S * sizeof(_Float16));
    IOSurfaceUnlock(ln->k3c->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
}
