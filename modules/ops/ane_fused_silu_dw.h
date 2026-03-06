// ane_fused_silu_dw.h — Fused silu + dW computation for pw2 backward
//
// Computes: dW_pw2 = dy @ silu(pw1_out)^T
//
// This replaces the need to store silu output during forward.
// Instead, silu is recomputed from pw1_out (pre-silu) inside the kernel.
//
// Inputs (ascending size for ANE slot rule):
//   slot 0: dy       [1, C, S]    — gradient from above
//   slot 1: pw1_out  [1, 4C, S]   — pw1 output (pre-silu activation)
//
// Output:
//   out 0:  dW_pw2   [1, C, 4C]   — weight gradient for pw2
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_FSD_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef dy_surf;       // slot 0: [C, S]
    IOSurfaceRef pw1_out_surf;  // slot 1: [4C, S]
    IOSurfaceRef dw_surf;       // output: [C, 4C]
} ANEFusedSiluDw;

static ANEFusedSiluDw *ane_fused_silu_dw_compile(int C, int S) {
    int C4 = C * 4;

    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_FSD_BI "\n{\n"
        "  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> dy,\n"
        "    tensor<fp16,[1,%d,%d]> pw1_out) {\n"
        "    tensor<fp16,[1,%d,%d]> sig=sigmoid(x=pw1_out)[name=tensor<string,[]>(\"sig\")];\n"
        "    tensor<fp16,[1,%d,%d]> sx=mul(x=pw1_out,y=sig)[name=tensor<string,[]>(\"sx\")];\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<fp16,[1,%d,%d]> dW=matmul(transpose_x=ff,transpose_y=tt,x=dy,y=sx)"
            "[name=tensor<string,[]>(\"dW\")];\n"
        "  } -> (dW);\n}\n",
        C, S,      // dy
        C4, S,     // pw1_out
        C4, S,     // sig
        C4, S,     // sx
        C, C4];    // dW

    size_t sdy = (size_t)C * S * 2;   if (sdy < 2048) sdy = 2048;
    size_t sx  = (size_t)C4 * S * 2;  if (sx < 2048) sx = 2048;
    size_t sdw = (size_t)C * C4 * 2;  if (sdw < 2048) sdw = 2048;

    size_t ins[2] = {sdy, sx};
    size_t outs[1] = {sdw};

    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins, 1, outs);
    if (!k) {
        fprintf(stderr, "ane_fused_silu_dw_compile FAILED (C=%d S=%d)\n", C, S);
        return NULL;
    }

    ANEFusedSiluDw *f = (ANEFusedSiluDw *)calloc(1, sizeof(ANEFusedSiluDw));
    f->C = C; f->S = S;
    f->k = k;
    f->dy_surf      = k->ioInputs[0];
    f->pw1_out_surf = k->ioInputs[1];
    f->dw_surf      = k->ioOutputs[0];
    return f;
}

static void ane_fused_silu_dw_eval(ANEFusedSiluDw *f) {
    ane_eval(f->k);
}
