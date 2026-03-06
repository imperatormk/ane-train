// ane_fused_pw_bwd.h — Fused pw2_dx + silu_bwd + pw1_dx in one ANE dispatch
//
// Computes the full dx chain for ConvNeXt backward pointwise path:
//   d_pw2  = W_pw2^T @ dy         (matmul dx, W_pw2[Co,4Co], dy[Co,S] → [4Co,S])
//   d_silu = d_pw2 * silu'(x)     (silu backward, x=silu_x[4Co,S])
//   dx     = W_pw1^T @ d_silu     (matmul dx, W_pw1[4Co,Ci], d_silu[4Co,S] → [Ci,S])
//
// Single ANE dispatch replaces 3 dispatches (pw2_dx + silu_bwd + pw1_dx).
// Outputs both dx AND d_silu (needed by pw1_dw kernel).
//
// Inputs (sorted ascending by size for ANE slot rule):
//   slot 0: W_pw2  [1, Co, 4*Co]
//   slot 1: W_pw1  [1, 4*Co, Ci]
//   slot 2: dy     [1, Co, S]
//   slot 3: silu_x [1, 4*Co, S]
//
// Outputs:
//   out 0: dx     [1, Ci, S]
//   out 1: d_silu [1, 4*Co, S]  (intermediate — needed for pw1 dW computation)
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_FPB_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int Ci, Co, S;
    ANEKernel *k;
    IOSurfaceRef w_pw2_surf;   // slot 0
    IOSurfaceRef w_pw1_surf;   // slot 1
    IOSurfaceRef dy_surf;      // slot 2
    IOSurfaceRef silu_x_surf;  // slot 3
    IOSurfaceRef dx_surf;      // output 0
    IOSurfaceRef d_silu_surf;  // output 1 (intermediate for pw1_dw)
} ANEFusedPwBwd;

static ANEFusedPwBwd *ane_fused_pw_bwd_compile(int Ci, int Co, int S) {
    int C4 = Co * 4;

    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_FPB_BI "\n{\n"
        "  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> W_pw2,\n"
        "    tensor<fp16,[1,%d,%d]> W_pw1,\n"
        "    tensor<fp16,[1,%d,%d]> dy,\n"
        "    tensor<fp16,[1,%d,%d]> silu_x) {\n"

        // Step 1: d_pw2 = W_pw2^T @ dy → [4Co, S]
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,%d,%d]> d_pw2=matmul(transpose_x=tt,transpose_y=ff,x=W_pw2,y=dy)"
            "[name=tensor<string,[]>(\"d_pw2\")];\n"

        // Step 2: silu_bwd
        "    tensor<fp16,[]> one=const()[name=tensor<string,[]>(\"one\"),val=tensor<fp16,[]>(1.0)];\n"
        "    tensor<fp16,[1,%d,%d]> sig =sigmoid(x=silu_x)[name=tensor<string,[]>(\"sig\")];\n"
        "    tensor<fp16,[1,%d,%d]> osig=sub(x=one,y=sig)[name=tensor<string,[]>(\"osig\")];\n"
        "    tensor<fp16,[1,%d,%d]> xosig=mul(x=silu_x,y=osig)[name=tensor<string,[]>(\"xosig\")];\n"
        "    tensor<fp16,[1,%d,%d]> ip1 =add(x=one,y=xosig)[name=tensor<string,[]>(\"ip1\")];\n"
        "    tensor<fp16,[1,%d,%d]> gpx =mul(x=sig,y=ip1)[name=tensor<string,[]>(\"gpx\")];\n"
        "    tensor<fp16,[1,%d,%d]> d_silu=mul(x=d_pw2,y=gpx)[name=tensor<string,[]>(\"d_silu\")];\n"

        // Step 3: dx = W_pw1^T @ d_silu → [Ci, S]
        "    tensor<fp16,[1,%d,%d]> dx=matmul(transpose_x=tt,transpose_y=ff,x=W_pw1,y=d_silu)"
            "[name=tensor<string,[]>(\"dx\")];\n"
        "  } -> (dx, d_silu);\n}\n",
        Co, C4,    // W_pw2
        C4, Ci,    // W_pw1
        Co, S,     // dy
        C4, S,     // silu_x
        C4, S,     // d_pw2
        C4, S,     // sig
        C4, S,     // osig
        C4, S,     // xosig
        C4, S,     // ip1
        C4, S,     // gpx
        C4, S,     // d_silu
        Ci, S];    // dx

    size_t sw1 = (size_t)Co * C4 * 2;   if (sw1 < 2048) sw1 = 2048;
    size_t sw2 = (size_t)C4 * Ci * 2;   if (sw2 < 2048) sw2 = 2048;
    size_t sdy = (size_t)Co * S  * 2;   if (sdy < 2048) sdy = 2048;
    size_t ssx = (size_t)C4 * S  * 2;   if (ssx < 2048) ssx = 2048;
    size_t sdx = (size_t)Ci * S  * 2;   if (sdx < 2048) sdx = 2048;
    size_t sds = (size_t)C4 * S  * 2;   if (sds < 2048) sds = 2048;

    size_t ins[4] = {sw1, sw2, sdy, ssx};
    size_t outs[2] = {sdx, sds};
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 4, ins, 2, outs);
    if (!k) {
        fprintf(stderr, "ane_fused_pw_bwd_compile FAILED (Ci=%d Co=%d S=%d)\n", Ci, Co, S);
        return NULL;
    }

    ANEFusedPwBwd *f = (ANEFusedPwBwd *)calloc(1, sizeof(ANEFusedPwBwd));
    f->Ci = Ci; f->Co = Co; f->S = S;
    f->k = k;
    f->w_pw2_surf  = k->ioInputs[0];
    f->w_pw1_surf  = k->ioInputs[1];
    f->dy_surf     = k->ioInputs[2];
    f->silu_x_surf = k->ioInputs[3];
    f->dx_surf     = k->ioOutputs[0];
    f->d_silu_surf = k->ioOutputs[1];
    return f;
}

static void ane_fused_pw_bwd_eval(ANEFusedPwBwd *f) {
    ane_eval(f->k);
}

static void ane_fused_pw_bwd_write_dy(ANEFusedPwBwd *f, const _Float16 *dy) {
    IOSurfaceLock(f->dy_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(f->dy_surf), dy, f->Co * f->S * sizeof(_Float16));
    IOSurfaceUnlock(f->dy_surf, 0, NULL);
}

static void ane_fused_pw_bwd_read_dx(ANEFusedPwBwd *f, _Float16 *dx) {
    IOSurfaceLock(f->dx_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(dx, IOSurfaceGetBaseAddress(f->dx_surf), f->Ci * f->S * sizeof(_Float16));
    IOSurfaceUnlock(f->dx_surf, kIOSurfaceLockReadOnly, NULL);
}
