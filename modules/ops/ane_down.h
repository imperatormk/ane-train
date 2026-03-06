// ane_down.h — Downsample: 2×2 stride-2 conv as im2col (CPU) + matmul (ANE)
//
// Input:  x[Cin, H, H] fp16
// Output: y[Cout, H/2, H/2] fp16
//
// im2col: non-overlapping 2×2 patches → col[Cin*4, S_out]  (Cin*4 always multiple of 4,
//   pad to multiple of 16 if needed)
// ANE matmul: W[Cout, Cin_pad] @ col[Cin_pad, S_out] → y[Cout, S_out]
//
// Usage:
//   ANEDown *dn = ane_down_compile(Cin, Cout, H);
//   ane_down_write_w(dn, w_fp16);   // [Cout, Cin_pad]
//   ane_down_eval(dn, x_fp16);      // [Cin*H*H] fp16
//   ane_down_read_y(dn, y_fp16);    // [Cout*S_out] fp16
//
#pragma once
#include "../../ane_runtime.h"
#include "ane_matmul.h"

typedef struct {
    int Cin, Cout, H;
    int Cin_pad;   // Cin*4 padded to multiple of 16
    int Ho;        // H/2
    int S_out;     // Ho*Ho
    ANEMatmul *mm;
    _Float16 *col_buf;  // [Cin_pad, S_out] fp16 scratch
} ANEDown;

static void _down_im2col(const _Float16 *x, _Float16 *col,
                          int Cin, int H, int Cin_pad, int Ho) {
    int K = 2, S_out = Ho * Ho;
    memset(col, 0, (size_t)Cin_pad * S_out * sizeof(_Float16));
    for (int c = 0; c < Cin; c++)
        for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            int row = c*K*K + ky*K + kx;
            for (int oh = 0; oh < Ho; oh++)
            for (int ow = 0; ow < Ho; ow++) {
                int ih = oh*K + ky, iw = ow*K + kx;
                col[row*S_out + oh*Ho + ow] = x[c*H*H + ih*H + iw];
            }
        }
}

static ANEDown *ane_down_compile(int Cin, int Cout, int H) {
    int raw = Cin * 4;  // 2×2 patches
    int Cin_pad = (raw + 31) & ~31;  // multiple of 32 (ANE matmul Ci constraint)
    int Ho = H / 2, S_out = Ho * Ho;

    ANEMatmul *mm = ane_matmul_compile(Cin_pad, Cout, S_out);
    if (!mm) {
        fprintf(stderr, "ane_down_compile FAILED (Cin=%d Cout=%d H=%d)\n", Cin, Cout, H);
        return NULL;
    }
    ANEDown *dn = (ANEDown *)calloc(1, sizeof(ANEDown));
    dn->Cin = Cin; dn->Cout = Cout; dn->H = H;
    dn->Cin_pad = Cin_pad; dn->Ho = Ho; dn->S_out = S_out;
    dn->mm = mm;
    dn->col_buf = (_Float16 *)calloc((size_t)Cin_pad * S_out, sizeof(_Float16));
    return dn;
}

static void ane_down_write_w(ANEDown *dn, const _Float16 *w) {
    ane_matmul_write_w(dn->mm, w);
}

static void ane_down_eval(ANEDown *dn, const _Float16 *x) {
    _down_im2col(x, dn->col_buf, dn->Cin, dn->H, dn->Cin_pad, dn->Ho);
    IOSurfaceLock(dn->mm->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(dn->mm->x_surf), dn->col_buf,
           (size_t)dn->Cin_pad * dn->S_out * sizeof(_Float16));
    IOSurfaceUnlock(dn->mm->x_surf, 0, NULL);
    ane_matmul_eval(dn->mm);
}

static void ane_down_read_y(ANEDown *dn, _Float16 *y) {
    ane_matmul_read_y(dn->mm, y);
}

static IOSurfaceRef ane_down_output_surf(ANEDown *dn) { return dn->mm->y_surf; }
