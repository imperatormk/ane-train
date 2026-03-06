// ane_stem_bwd.h — Stem backward: col2im (CPU) + matmul bwd (ANE)
//
// Forward: im2col(x[Cin,H,H], stride=2) → col[Cin_pad,S_out] → W[Cout,Cin_pad] @ col → y[Cout,S_out]
// Backward:
//   dW[Cout,Cin_pad] = dy[Cout,S_out] @ col^T       (ANE matmul_bwd dW)
//   d_col[Cin_pad,S_out] = W^T @ dy                 (ANE matmul_bwd dX)
//   dx[Cin,H,H]      = col2im(d_col)                (CPU — inverse of im2col)
//
// col2im for K×K stride-2 patches with zero-padding:
//   dx[c, ih, iw] += d_col[c*K*K+ky*K+kx, oh*Ho+ow]
//   for all (ky,kx,oh,ow) where ih=oh*2+ky-pad, iw=ow*2+kx-pad
//
#pragma once
#include "ane_stem.h"
#include "ane_matmul_bwd.h"

typedef struct {
    ANEStem      *fwd;
    ANEMatmulBwd *mm_bwd;
    _Float16     *d_col;   // [Cin_pad, S_out] scratch
} ANEStemBwd;

static void _stem_col2im(const _Float16 *d_col, _Float16 *dx,
                          int Cin, int H, int K, int Cin_pad, int Ho) {
    int stride = 2, pad = (K - stride) / 2;
    int S_out = Ho * Ho;
    memset(dx, 0, (size_t)Cin * H * H * sizeof(_Float16));
    for (int c = 0; c < Cin; c++)
        for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            int row = c*K*K + ky*K + kx;
            for (int oh = 0; oh < Ho; oh++)
            for (int ow = 0; ow < Ho; ow++) {
                int ih = oh*stride + ky - pad, iw = ow*stride + kx - pad;
                if (ih >= 0 && ih < H && iw >= 0 && iw < H)
                    dx[c*H*H + ih*H + iw] += d_col[row*S_out + oh*Ho + ow];
            }
        }
}

static ANEStemBwd *ane_stem_bwd_compile(ANEStem *fwd) {
    ANEMatmulBwd *mm_bwd = ane_matmul_bwd_compile(fwd->Cin_pad, fwd->Cout, fwd->S_out);
    if (!mm_bwd) { fprintf(stderr, "ane_stem_bwd_compile FAILED\n"); return NULL; }
    // Share W with forward
    ane_matmul_bwd_rewire_w(mm_bwd, fwd->mm->w_surf);
    // Share col (x) with forward's mm->x_surf (im2col output written there each fwd pass)
    ane_matmul_bwd_rewire_x(mm_bwd, fwd->mm->x_surf);

    ANEStemBwd *bwd = (ANEStemBwd *)calloc(1, sizeof(ANEStemBwd));
    bwd->fwd    = fwd;
    bwd->mm_bwd = mm_bwd;
    bwd->d_col  = (_Float16 *)malloc((size_t)fwd->Cin_pad * fwd->S_out * sizeof(_Float16));
    return bwd;
}

// dy[Cout,S_out], dx[Cin,H,H] output
static void ane_stem_bwd_eval(ANEStemBwd *bwd, const _Float16 *dy, _Float16 *dx) {
    ANEStem *f = bwd->fwd;
    ane_matmul_bwd_write_dy(bwd->mm_bwd, dy);
    ane_matmul_bwd_eval(bwd->mm_bwd);
    // Read d_col
    ane_matmul_bwd_read_dx(bwd->mm_bwd, bwd->d_col);
    // col2im → dx
    _stem_col2im(bwd->d_col, dx, f->Cin, f->H, f->K, f->Cin_pad, f->Ho);
}

static void ane_stem_bwd_read_dw(ANEStemBwd *bwd, _Float16 *dw) {
    ane_matmul_bwd_read_dw(bwd->mm_bwd, dw);
}
