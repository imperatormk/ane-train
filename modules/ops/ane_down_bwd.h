// ane_down_bwd.h — Down (2×2 stride-2 conv) backward: col2im + matmul bwd (ANE)
//
// Identical structure to ane_stem_bwd.h but K=2 fixed.
//
#pragma once
#include "ane_down.h"
#include "ane_matmul_bwd.h"

typedef struct {
    ANEDown      *fwd;
    ANEMatmulBwd *mm_bwd;
    _Float16     *d_col;   // [Cin_pad, S_out]
} ANEDownBwd;

static void _down_col2im(const _Float16 *d_col, _Float16 *dx,
                          int Cin, int H, int Cin_pad, int Ho) {
    int K = 2, S_out = Ho * Ho;
    memset(dx, 0, (size_t)Cin * H * H * sizeof(_Float16));
    for (int c = 0; c < Cin; c++)
        for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            int row = c*K*K + ky*K + kx;
            for (int oh = 0; oh < Ho; oh++)
            for (int ow = 0; ow < Ho; ow++) {
                int ih = oh*K + ky, iw = ow*K + kx;
                dx[c*H*H + ih*H + iw] = d_col[row*S_out + oh*Ho + ow];
            }
        }
}

static ANEDownBwd *ane_down_bwd_compile(ANEDown *fwd) {
    ANEMatmulBwd *mm_bwd = ane_matmul_bwd_compile(fwd->Cin_pad, fwd->Cout, fwd->S_out);
    if (!mm_bwd) { fprintf(stderr, "ane_down_bwd_compile FAILED\n"); return NULL; }
    ane_matmul_bwd_rewire_w(mm_bwd, fwd->mm->w_surf);
    ane_matmul_bwd_rewire_x(mm_bwd, fwd->mm->x_surf);

    ANEDownBwd *bwd = (ANEDownBwd *)calloc(1, sizeof(ANEDownBwd));
    bwd->fwd    = fwd;
    bwd->mm_bwd = mm_bwd;
    bwd->d_col  = (_Float16 *)malloc((size_t)fwd->Cin_pad * fwd->S_out * sizeof(_Float16));
    return bwd;
}

// dy[Cout,S_out], dx[Cin,H,H] output
static void ane_down_bwd_eval(ANEDownBwd *bwd, const _Float16 *dy, _Float16 *dx) {
    ANEDown *f = bwd->fwd;
    ane_matmul_bwd_write_dy(bwd->mm_bwd, dy);
    ane_matmul_bwd_eval(bwd->mm_bwd);
    ane_matmul_bwd_read_dx(bwd->mm_bwd, bwd->d_col);
    _down_col2im(bwd->d_col, dx, f->Cin, f->H, f->Cin_pad, f->Ho);
}

static void ane_down_bwd_read_dw(ANEDownBwd *bwd, _Float16 *dw) {
    ane_matmul_bwd_read_dw(bwd->mm_bwd, dw);
}
