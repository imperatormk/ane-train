// ane_fused_silu_pw2_add.h — Fused SiLU + pw2 matmul + residual add (1 dispatch)
//
// Computes: y = W_pw2 @ silu(x) + residual
//
// Inputs (ascending size for ANE slot rule):
//   slot 0: W_pw2     [1, C, 4C]   — pw2 weight matrix
//   slot 1: residual  [1, C, S]    — skip connection input
//   slot 2: x         [1, 4C, S]   — pw1 output (pre-silu)
//
// Output:
//   out 0:  y         [1, C, S]    — block output
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_FSPA_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef w_surf;         // slot 0: [C, 4C]
    IOSurfaceRef residual_surf;  // slot 1: [C, S]
    IOSurfaceRef x_surf;         // slot 2: [4C, S]
    IOSurfaceRef y_surf;         // output: [C, S]
} ANEFusedSiluPw2Add;

static ANEFusedSiluPw2Add *ane_fused_silu_pw2_add_compile(int C, int S) {
    int C4 = C * 4;

    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_FSPA_BI "\n{\n"
        "  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> W,\n"
        "    tensor<fp16,[1,%d,%d]> residual,\n"
        "    tensor<fp16,[1,%d,%d]> x) {\n"
        "    tensor<fp16,[1,%d,%d]> sig=sigmoid(x=x)[name=tensor<string,[]>(\"sig\")];\n"
        "    tensor<fp16,[1,%d,%d]> sx=mul(x=x,y=sig)[name=tensor<string,[]>(\"sx\")];\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,%d,%d]> mm=matmul(transpose_x=ff,transpose_y=ff,x=W,y=sx)"
            "[name=tensor<string,[]>(\"mm\")];\n"
        "    tensor<fp16,[1,%d,%d]> y=add(x=mm,y=residual)"
            "[name=tensor<string,[]>(\"y\")];\n"
        "  } -> (y);\n}\n",
        C, C4,     // W
        C, S,      // residual
        C4, S,     // x
        C4, S,     // sig
        C4, S,     // sx
        C, S,      // mm
        C, S];     // y

    size_t sw = (size_t)C * C4 * 2;  if (sw < 2048) sw = 2048;
    size_t sr = (size_t)C * S * 2;   if (sr < 2048) sr = 2048;
    size_t sx = (size_t)C4 * S * 2;  if (sx < 2048) sx = 2048;
    size_t sy = (size_t)C * S * 2;   if (sy < 2048) sy = 2048;

    size_t ins[3] = {sw, sr, sx};
    size_t outs[1] = {sy};

    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 3, ins, 1, outs);
    if (!k) {
        fprintf(stderr, "ane_fused_silu_pw2_add_compile FAILED (C=%d S=%d)\n", C, S);
        return NULL;
    }

    ANEFusedSiluPw2Add *f = (ANEFusedSiluPw2Add *)calloc(1, sizeof(ANEFusedSiluPw2Add));
    f->C = C; f->S = S;
    f->k = k;
    f->w_surf        = k->ioInputs[0];
    f->residual_surf = k->ioInputs[1];
    f->x_surf        = k->ioInputs[2];
    f->y_surf        = k->ioOutputs[0];
    return f;
}

static void ane_fused_silu_pw2_add_eval(ANEFusedSiluPw2Add *f) {
    ane_eval(f->k);
}
