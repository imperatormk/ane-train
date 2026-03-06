// ane_dw_bwd.h — Depthwise conv KxK backward (CPU NEON)
//
// Forward:  y[c,oh,ow] = sum_{ky,kx} w[c,ky,kx] * x[c, oh+ky-pad, ow+kx-pad]
// Backward:
//   dx[c,ih,iw] += sum_{ky,kx} w[c,ky,kx] * dy[c, ih-(ky-pad), iw-(kx-pad)]
//               (full conv of dy with flipped kernel)
//   dw[c,ky,kx]  = sum_{oh,ow} dy[c,oh,ow] * x[c, oh+ky-pad, ow+kx-pad]
//
// Layout: x, y, dx, dy all [C, H, H] (C-major, S=H*H spatial)
// w, dw: [C, K*K]
//
// NEON vectorized over spatial W (8 pixels at a time), same pattern as fwd.
//
#pragma once
#include <arm_neon.h>
#include <string.h>
#include <dispatch/dispatch.h>

// Backward dx: full conv with flipped kernel. Row-output-first for L1 cache locality.
// Parallel over C (channels independent).
static void __attribute__((noinline))
_dw_neon_bwd_dx(const _Float16 *dy, const _Float16 *w, _Float16 *dx,
                int C, int H, int K) {
    int S = H * H, pad = K / 2;
    dispatch_apply((size_t)C, DISPATCH_APPLY_AUTO, ^(size_t c_) {
        int c = (int)c_;
        const _Float16 *dyc = dy + c*S;
        _Float16 *dxc = dx + c*S;
        float acc_row[H];
        // Pre-load flipped weights
        float wf[K*K];
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                wf[i*K+j] = (float)w[c*K*K + (K-1-i)*K + (K-1-j)];

        for (int ih = 0; ih < H; ih++) {
            memset(acc_row, 0, (size_t)H * sizeof(float));
            for (int ky = 0; ky < K; ky++) {
                int oh = ih - (ky - pad);
                if (oh < 0 || oh >= H) continue;
                const _Float16 *dyrow = dyc + oh*H;
                for (int kx = 0; kx < K; kx++) {
                    float wv = wf[ky*K + kx];
                    float32x4_t vw = vdupq_n_f32(wv);
                    int dx_shift = kx - pad;
                    int iw_start = (dx_shift > 0) ? dx_shift : 0;
                    int iw_end   = (H + dx_shift < H) ? H + dx_shift : H;
                    int iw = iw_start;
                    for (; iw + 7 < iw_end; iw += 8) {
                        int ow = iw - dx_shift;
                        float16x8_t dyh = vld1q_f16((const __fp16*)(dyrow + ow));
                        float32x4_t dyl = vcvt_f32_f16(vget_low_f16(dyh));
                        float32x4_t dyhi = vcvt_f32_f16(vget_high_f16(dyh));
                        vst1q_f32(acc_row+iw,   vfmaq_f32(vld1q_f32(acc_row+iw),   dyl,  vw));
                        vst1q_f32(acc_row+iw+4, vfmaq_f32(vld1q_f32(acc_row+iw+4), dyhi, vw));
                    }
                    for (; iw < iw_end; iw++) {
                        int ow = iw - dx_shift;
                        acc_row[iw] += wv * (float)dyrow[ow];
                    }
                }
            }
            _Float16 *dxr = dxc + ih*H;
            int iw = 0;
            for (; iw + 7 < H; iw += 8)
                vst1q_f16((__fp16*)(dxr+iw),
                    vcombine_f16(vcvt_f16_f32(vld1q_f32(acc_row+iw)),
                                 vcvt_f16_f32(vld1q_f32(acc_row+iw+4))));
            for (; iw < H; iw++) dxr[iw] = (_Float16)acc_row[iw];
        }
    });
}

// Backward dw: dw[c,ky,kx] = sum_{oh,ow} dy*x. Parallel over C.
static void __attribute__((noinline))
_dw_neon_bwd_dw(const _Float16 *dy, const _Float16 *x, _Float16 *dw,
                int C, int H, int K) {
    int S = H * H, pad = K / 2;
    dispatch_apply((size_t)C, DISPATCH_APPLY_AUTO, ^(size_t c_) {
        int c = (int)c_;
        const _Float16 *dyc = dy + c*S;
        const _Float16 *xc  = x  + c*S;
        for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            float acc = 0.0f;
            int dkx = kx - pad;
            for (int oh = 0; oh < H; oh++) {
                int ih = oh + ky - pad;
                if (ih < 0 || ih >= H) continue;
                int ow_start = (-dkx > 0) ? -dkx : 0;
                int ow_end   = (H - dkx < H) ? H - dkx : H;
                int ow = ow_start;
                float32x4_t vacc0 = vdupq_n_f32(0.0f);
                float32x4_t vacc1 = vdupq_n_f32(0.0f);
                for (; ow + 7 < ow_end; ow += 8) {
                    int iw = ow + dkx;
                    float16x8_t dyh = vld1q_f16((const __fp16*)(dyc + oh*H + ow));
                    float16x8_t xh  = vld1q_f16((const __fp16*)(xc  + ih*H + iw));
                    vacc0 = vfmaq_f32(vacc0, vcvt_f32_f16(vget_low_f16(dyh)),
                                             vcvt_f32_f16(vget_low_f16(xh)));
                    vacc1 = vfmaq_f32(vacc1, vcvt_f32_f16(vget_high_f16(dyh)),
                                             vcvt_f32_f16(vget_high_f16(xh)));
                }
                acc += vaddvq_f32(vaddq_f32(vacc0, vacc1));
                for (; ow < ow_end; ow++) {
                    int iw = ow + dkx;
                    acc += (float)dyc[oh*H + ow] * (float)xc[ih*H + iw];
                }
            }
            dw[c*K*K + ky*K + kx] = (_Float16)acc;
        }
    });
}
