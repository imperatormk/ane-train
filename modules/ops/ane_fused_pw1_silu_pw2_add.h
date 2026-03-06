// ane_fused_pw1_silu_pw2_add.h — Fused pw1 + SiLU + pw2 + residual add (1 dispatch)
//
// Computes: pw1_out = W_pw1 @ x
//           y = W_pw2 @ silu(pw1_out) + residual
//
// Replaces: pw1 matmul + silu + pw2 matmul + add = 4 dispatches → 1
//
// Inputs (ascending size for ANE slot rule):
//   slot 0: W_pw1     [1, 4C, C]   — pw1 weight matrix
//   slot 1: W_pw2     [1, C, 4C]   — pw2 weight matrix  (same bytes as W_pw1)
//   slot 2: residual  [1, C, S]    — skip connection input
//   slot 3: x         [1, C, S]    — LN2 output (same bytes as residual)
//
// Outputs (descending size for ANE slot rule):
//   out 0:  pw1_out   [1, 4C, S]   — pw1 output (pre-silu), needed by backward
//   out 1:  y         [1, C, S]    — block output
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_FP1SPA_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef w_pw1_surf;     // slot 0: [4C, C]
    IOSurfaceRef w_pw2_surf;     // slot 1: [C, 4C]
    IOSurfaceRef residual_surf;  // slot 2: [C, S]
    IOSurfaceRef x_surf;         // slot 3: [C, S]
    IOSurfaceRef pw1_out_surf;   // output 0: [4C, S]
    IOSurfaceRef y_surf;         // output 1: [C, S]
} ANEFusedPw1SiluPw2Add;

static ANEFusedPw1SiluPw2Add *ane_fused_pw1_silu_pw2_add_compile(int C, int S) {
    int C4 = C * 4;

    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_FP1SPA_BI "\n{\n"
        "  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> W_pw1,\n"
        "    tensor<fp16,[1,%d,%d]> W_pw2,\n"
        "    tensor<fp16,[1,%d,%d]> residual,\n"
        "    tensor<fp16,[1,%d,%d]> x) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,%d,%d]> pw1_out=matmul(transpose_x=ff,transpose_y=ff,x=W_pw1,y=x)"
            "[name=tensor<string,[]>(\"pw1_out\")];\n"
        "    tensor<fp16,[1,%d,%d]> sig=sigmoid(x=pw1_out)[name=tensor<string,[]>(\"sig\")];\n"
        "    tensor<fp16,[1,%d,%d]> sx=mul(x=pw1_out,y=sig)[name=tensor<string,[]>(\"sx\")];\n"
        "    tensor<fp16,[1,%d,%d]> mm=matmul(transpose_x=ff,transpose_y=ff,x=W_pw2,y=sx)"
            "[name=tensor<string,[]>(\"mm\")];\n"
        "    tensor<fp16,[1,%d,%d]> y=add(x=mm,y=residual)"
            "[name=tensor<string,[]>(\"y\")];\n"
        "  } -> (pw1_out, y);\n}\n",
        C4, C,     // W_pw1
        C, C4,     // W_pw2
        C, S,      // residual
        C, S,      // x
        C4, S,     // pw1_out
        C4, S,     // sig
        C4, S,     // sx
        C, S,      // mm
        C, S];     // y

    size_t sw1 = (size_t)C4 * C * 2;  if (sw1 < 2048) sw1 = 2048;
    size_t sw2 = (size_t)C * C4 * 2;  if (sw2 < 2048) sw2 = 2048;
    size_t sr  = (size_t)C * S * 2;   if (sr < 2048) sr = 2048;
    size_t sx  = (size_t)C * S * 2;   if (sx < 2048) sx = 2048;
    size_t sp  = (size_t)C4 * S * 2;  if (sp < 2048) sp = 2048;
    size_t sy  = (size_t)C * S * 2;   if (sy < 2048) sy = 2048;

    size_t ins[4] = {sw1, sw2, sr, sx};
    size_t outs[2] = {sp, sy};  // descending: pw1_out[4C,S] > y[C,S]

    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 4, ins, 2, outs);
    if (!k) {
        fprintf(stderr, "ane_fused_pw1_silu_pw2_add_compile FAILED (C=%d S=%d)\n", C, S);
        return NULL;
    }

    ANEFusedPw1SiluPw2Add *f = (ANEFusedPw1SiluPw2Add *)calloc(1, sizeof(ANEFusedPw1SiluPw2Add));
    f->C = C; f->S = S;
    f->k = k;
    f->w_pw1_surf     = k->ioInputs[0];
    f->w_pw2_surf     = k->ioInputs[1];
    f->residual_surf  = k->ioInputs[2];
    f->x_surf         = k->ioInputs[3];
    f->pw1_out_surf   = k->ioOutputs[0];
    f->y_surf         = k->ioOutputs[1];
    return f;
}

static void ane_fused_pw1_silu_pw2_add_eval(ANEFusedPw1SiluPw2Add *f) {
    ane_eval(f->k);
}
