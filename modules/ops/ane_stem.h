// ane_stem.h — Stem: K×K stride-2 conv as im2col (CPU) + matmul (ANE)
//
// Input:  x[Cin, H, H] fp16
// Output: y[Cout, H/2, H/2] fp16   (S_out = (H/2)^2)
//
// im2col packs overlapping K×K patches with stride=2 → col[Cin*K*K, S_out]
// then ANE matmul: W[Cout, Cin_pad] @ col[Cin_pad, S_out] → y[Cout, S_out]
//
// Cin*K*K may not be a multiple of 32 — pad to next multiple of 32 (ANE matmul Ci constraint).
// Padding rows in col are zero; corresponding W rows are also zero (no effect).
// Out-of-bounds reads (ih/iw < 0 or >= H) are zero-padded.
//
// Usage:
//   ANEStem *stem = ane_stem_compile(Cin, Cout, H, K);  // K=4 for stem
//   ane_stem_write_w(stem, w_fp16);   // [Cout, Cin_pad]
//   ane_stem_eval(stem, x_fp16);      // [Cin*H*H] fp16 in
//   ane_stem_read_y(stem, y_fp16);    // [Cout*S_out] fp16 out
//
#pragma once
#include "../../ane_runtime.h"
#include "ane_matmul.h"
#include <arm_neon.h>

typedef struct {
    int Cin, Cout, H, K;
    int Cin_pad;   // Cin*K*K padded to multiple of 32
    int Ho;        // H/2 (output spatial side, stride=2)
    int S_out;     // Ho*Ho
    ANEMatmul *mm;
    _Float16 *col_buf;  // [Cin_pad, S_out] fp16 scratch
} ANEStem;

// im2col: K×K patches with stride=2, zero-padded, fp16 in/out
// Input x[Cin, H, H], output col[Cin_pad, S_out] (extra rows zeroed)
static void _stem_im2col(const _Float16 *x, _Float16 *col,
                          int Cin, int H, int K, int Cin_pad, int Ho) {
    int stride = 2, pad = (K - stride) / 2;  // pad=1 for K=4 s2
    int S_out = Ho * Ho;
    memset(col, 0, (size_t)Cin_pad * S_out * sizeof(_Float16));
    for (int c = 0; c < Cin; c++)
        for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            int row = c*K*K + ky*K + kx;
            for (int oh = 0; oh < Ho; oh++)
            for (int ow = 0; ow < Ho; ow++) {
                int ih = oh*stride + ky - pad, iw = ow*stride + kx - pad;
                if (ih >= 0 && ih < H && iw >= 0 && iw < H)
                    col[row*S_out + oh*Ho + ow] = x[c*H*H + ih*H + iw];
            }
        }
}

static ANEStem *ane_stem_compile(int Cin, int Cout, int H, int K) {
    int KK = K * K;
    int raw = Cin * KK;
    int Cin_pad = (raw + 31) & ~31;  // round up to multiple of 32 (ANE matmul Ci constraint)
    int Ho = H / 2;  // stride=2
    int S_out = Ho * Ho;

    ANEMatmul *mm = ane_matmul_compile(Cin_pad, Cout, S_out);
    if (!mm) {
        fprintf(stderr, "ane_stem_compile FAILED (Cin=%d Cout=%d H=%d K=%d)\n", Cin, Cout, H, K);
        return NULL;
    }
    ANEStem *stem = (ANEStem *)calloc(1, sizeof(ANEStem));
    stem->Cin = Cin; stem->Cout = Cout; stem->H = H; stem->K = K;
    stem->Cin_pad = Cin_pad; stem->Ho = Ho; stem->S_out = S_out;
    stem->mm = mm;
    stem->col_buf = (_Float16 *)calloc((size_t)Cin_pad * S_out, sizeof(_Float16));
    return stem;
}

// w: [Cout, Cin_pad] row-major fp16
static void ane_stem_write_w(ANEStem *stem, const _Float16 *w) {
    ane_matmul_write_w(stem->mm, w);
}

// x: [Cin, H, H] fp16
static void ane_stem_eval(ANEStem *stem, const _Float16 *x) {
    _stem_im2col(x, stem->col_buf, stem->Cin, stem->H, stem->K, stem->Cin_pad, stem->Ho);
    IOSurfaceLock(stem->mm->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(stem->mm->x_surf), stem->col_buf,
           (size_t)stem->Cin_pad * stem->S_out * sizeof(_Float16));
    IOSurfaceUnlock(stem->mm->x_surf, 0, NULL);
    ane_matmul_eval(stem->mm);
}

// y: [Cout, S_out] fp16
static void ane_stem_read_y(ANEStem *stem, _Float16 *y) {
    ane_matmul_read_y(stem->mm, y);
}

static IOSurfaceRef ane_stem_output_surf(ANEStem *stem) { return stem->mm->y_surf; }
