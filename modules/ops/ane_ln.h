// ane_ln.h — Batched LayerNorm on ANE
//
// Computes LN over channel dim: y[c,s] = (x[c,s] - mean_s) / sqrt(var_s + eps)
// Layout: [1, C, S] — C channels, S spatial/token positions
// All ANE, zero CPU roundtrips after compile.
//
// Usage:
//   ANELayerNorm *ln = ane_ln_compile(C, S);
//   ane_ln_write_input(ln, x_fp16_ptr);   // write [C*S] fp16
//   ane_ln_eval(ln);
//   ane_ln_read_output(ln, y_fp16_ptr);   // read [C*S] fp16
//   // or chain: ln->out is an IOSurfaceRef you can rewire into next kernel
//
#pragma once
#include "../../ane_runtime.h"
#include "../../mil_gen.h"
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ANE_LN_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k1a;     // matmul → mean [1,1,S]
    ANEKernel *k1b_2_3a; // fused: sub+sq+matmul → DIFF [1,C,S] + VAR [1,1,S]
    ANEKernel *k3b;     // add(VAR,EPS)+sqrt → SQRTV [1,1,S]
    ANEKernel *k3c;     // real_div(SQRTV,ONE)=1/sqrt → RSTD [1,1,S]
    ANEKernel *k4;      // mul(DIFF,RSTD) → NORM [1,C,S]
    IOSurfaceRef inp;   // input surface (k1a->ioInputs[1])
    IOSurfaceRef out;   // output surface (k4->ioOutputs[0])
} ANELayerNorm;

static void _ane_ln_rewire_in(ANEKernel *k, int slot, IOSurfaceRef surf) {
    IOSurfaceRef ins[16] = {0};
    ins[slot] = surf;
    ane_rewire(k, ins, NULL);
}

static ANELayerNorm *ane_ln_compile(int C, int S) {
    float eps = 1e-5f;
    size_t sw = (size_t)C * 2;       if (sw < 2048) sw = 2048;
    size_t si = (size_t)C * S * 2;
    size_t sv = (size_t)S * 2;       if (sv < 2048) sv = 2048;

    // k1a: mean matmul [1,C] @ [C,S] → [1,S]
    NSString *m1a = mil_gen_matmul_fwd(C, 1, S);

    // k1b_2_3a: fused sub + square + var matmul (3 dispatches → 1)
    // Inputs: INP[C,S], MN[1,S], W_var[1,C]
    // Outputs: DIFF[C,S], VAR[1,S]
    NSString *m_fused = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_LN_BI "\n{\n  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> INP,\n    tensor<fp16,[1,1,%d]> MN,\n"
        "    tensor<fp16,[1,1,%d]> W) {\n"
        "    tensor<fp16,[1,%d,%d]> DIFF=sub(x=INP,y=MN)[name=tensor<string,[]>(\"DIFF\")];\n"
        "    tensor<fp16,[1,%d,%d]> SQ=mul(x=DIFF,y=DIFF)[name=tensor<string,[]>(\"SQ\")];\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,1,%d]> VAR=matmul(transpose_x=ff,transpose_y=ff,x=W,y=SQ)"
        "[name=tensor<string,[]>(\"VAR\")];\n"
        "  } -> (DIFF, VAR);\n}\n", C, S, S, C, C, S, C, S, S];

    NSString *m3b = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_LN_BI "\n{\n  func main<ios16>(\n"
        "    tensor<fp16,[1,1,%d]> VAR,\n    tensor<fp16,[1,1,%d]> EPS) {\n"
        "    tensor<fp16,[1,1,%d]> VEPS=add(x=VAR,y=EPS)[name=tensor<string,[]>(\"VEPS\")];\n"
        "    tensor<fp16,[1,1,%d]> SQRTV=sqrt(x=VEPS)[name=tensor<string,[]>(\"SQRTV\")];\n"
        "  } -> (SQRTV);\n}\n", S, S, S, S];
    NSString *m3c = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_LN_BI "\n{\n  func main<ios16>(\n"
        "    tensor<fp16,[1,1,%d]> SQRTV,\n    tensor<fp16,[1,1,%d]> ONE) {\n"
        "    tensor<fp16,[1,1,%d]> RSTD=real_div(x=SQRTV,y=ONE)[name=tensor<string,[]>(\"RSTD\")];\n"
        "  } -> (RSTD);\n}\n", S, S, S];
    NSString *m4 = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_LN_BI "\n{\n  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> DIFF,\n    tensor<fp16,[1,1,%d]> RSTD) {\n"
        "    tensor<fp16,[1,%d,%d]> NORM=mul(x=DIFF,y=RSTD)[name=tensor<string,[]>(\"NORM\")];\n"
        "  } -> (NORM);\n}\n", C, S, S, C, S];

    // Fused kernel: 3 inputs (ascending: INP[C,S] ≥ MN[1,S], but sw < sv < si)
    // Input order: INP[si], MN[sv], W[sw] — must be ascending
    // si > sv > sw, so order: W[sw], MN[sv], INP[si]... but MIL param order matters
    // ANE slot rule: ioInputs[0].bytes ≤ ioInputs[1].bytes ≤ ...
    // So: slot0=W[sw], slot1=MN[sv], slot2=INP[si] — ascending ✓
    // But MIL param order = slot order, so reorder MIL params
    // Wait — our MIL has INP first, MN second, W third. That's si, sv, sw = DESCENDING. Wrong!
    // Need to reorder: W first, MN second, INP third
    NSString *m_fused_ordered = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_LN_BI "\n{\n  func main<ios16>(\n"
        "    tensor<fp16,[1,1,%d]> W,\n    tensor<fp16,[1,1,%d]> MN,\n"
        "    tensor<fp16,[1,%d,%d]> INP) {\n"
        "    tensor<fp16,[1,%d,%d]> DIFF=sub(x=INP,y=MN)[name=tensor<string,[]>(\"DIFF\")];\n"
        "    tensor<fp16,[1,%d,%d]> SQ=mul(x=DIFF,y=DIFF)[name=tensor<string,[]>(\"SQ\")];\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,1,%d]> VAR=matmul(transpose_x=ff,transpose_y=ff,x=W,y=SQ)"
        "[name=tensor<string,[]>(\"VAR\")];\n"
        "  } -> (DIFF, VAR);\n}\n", C, S, C, S, C, S, C, S, S];

    size_t ins1a[2] = {sw, si};
    size_t ins_fused[3] = {sw, sv, si};  // ascending: W, MN, INP
    size_t outs_fused[2] = {si, sv};     // descending: DIFF[C,S], VAR[1,S]
    size_t ins3b[2] = {sv, sv}, ins3c[2] = {sv, sv}, ins4[2] = {si, sv};

    ANELayerNorm *ln = (ANELayerNorm *)calloc(1, sizeof(ANELayerNorm));
    ln->C = C; ln->S = S;
    ln->k1a = ane_compile([m1a dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins1a, 1, &sv);
    ln->k1b_2_3a = ane_compile([m_fused_ordered dataUsingEncoding:NSUTF8StringEncoding], nil,
                                3, ins_fused, 2, outs_fused);
    ln->k3b = ane_compile([m3b dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins3b, 1, &sv);
    ln->k3c = ane_compile([m3c dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins3c, 1, &sv);
    ln->k4  = ane_compile([m4  dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins4,  1, &si);

    if (!ln->k1a||!ln->k1b_2_3a||!ln->k3b||!ln->k3c||!ln->k4) {
        fprintf(stderr, "ane_ln_compile FAILED (C=%d S=%d)\n", C, S);
        free(ln); return NULL;
    }

    // Bake mean weights (1/C each) for k1a
    IOSurfaceLock(ln->k1a->ioInputs[0], 0, NULL);
    for (int i = 0; i < C; i++) ((_Float16*)IOSurfaceGetBaseAddress(ln->k1a->ioInputs[0]))[i] = 1.0f/C;
    IOSurfaceUnlock(ln->k1a->ioInputs[0], 0, NULL);

    // Bake var weights (1/C each) for fused kernel slot 0
    IOSurfaceLock(ln->k1b_2_3a->ioInputs[0], 0, NULL);
    for (int i = 0; i < C; i++) ((_Float16*)IOSurfaceGetBaseAddress(ln->k1b_2_3a->ioInputs[0]))[i] = 1.0f/C;
    IOSurfaceUnlock(ln->k1b_2_3a->ioInputs[0], 0, NULL);

    // Bake eps and 1.0 constants
    IOSurfaceLock(ln->k3b->ioInputs[1], 0, NULL);
    for (int i = 0; i < S; i++) ((_Float16*)IOSurfaceGetBaseAddress(ln->k3b->ioInputs[1]))[i] = eps;
    IOSurfaceUnlock(ln->k3b->ioInputs[1], 0, NULL);
    IOSurfaceLock(ln->k3c->ioInputs[1], 0, NULL);
    for (int i = 0; i < S; i++) ((_Float16*)IOSurfaceGetBaseAddress(ln->k3c->ioInputs[1]))[i] = 1.0f;
    IOSurfaceUnlock(ln->k3c->ioInputs[1], 0, NULL);

    // Wire pipeline: k1a → fused(sub+sq+matmul) → k3b → k3c → k4
    // Fused kernel: slot0=W(baked), slot1=MN(from k1a), slot2=INP(input)
    _ane_ln_rewire_in(ln->k1b_2_3a, 1, ln->k1a->ioOutputs[0]);  // MN → fused slot1
    _ane_ln_rewire_in(ln->k1b_2_3a, 2, ln->k1a->ioInputs[1]);   // INP → fused slot2
    // Fused outputs: [0]=DIFF[C,S], [1]=VAR[1,S]
    _ane_ln_rewire_in(ln->k3b, 0, ln->k1b_2_3a->ioOutputs[1]);  // VAR → k3b
    _ane_ln_rewire_in(ln->k3c, 0, ln->k3b->ioOutputs[0]);        // SQRTV → k3c
    _ane_ln_rewire_in(ln->k4,  0, ln->k1b_2_3a->ioOutputs[0]);  // DIFF → k4 slot0
    _ane_ln_rewire_in(ln->k4,  1, ln->k3c->ioOutputs[0]);        // RSTD → k4 slot1

    ln->inp = ln->k1a->ioInputs[1];   // write here to set input
    ln->out = ln->k4->ioOutputs[0];   // read here after eval

    return ln;
}

// Rewire ln1 input to an external IOSurface (for block-to-block chaining).
// Must rewire k1a->slot1 (for mean) and fused->slot2 (INP for sub).
static void ane_ln_rewire_input(ANELayerNorm *ln, IOSurfaceRef surf) {
    _ane_ln_rewire_in(ln->k1a, 1, surf);
    _ane_ln_rewire_in(ln->k1b_2_3a, 2, surf);
    ln->inp = surf;
}

static void ane_ln_write_input(ANELayerNorm *ln, const _Float16 *x) {
    IOSurfaceLock(ln->inp, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(ln->inp), x, ln->C * ln->S * sizeof(_Float16));
    IOSurfaceUnlock(ln->inp, 0, NULL);
}

// CPU NEON LayerNorm: normalizes x[C,S] per-token (over C dim).
// Writes norm[C,S] to ln->out and rstd[S] to ln->k3c->ioOutputs[0]
// so backward (ane_ln_save_stats) still works correctly.
static void _ane_ln_eval_cpu(ANELayerNorm *ln) {
    int C = ln->C, S = ln->S;
    float inv_C = 1.0f / (float)C;
    float eps = 1e-5f;

    // Read input
    IOSurfaceLock(ln->inp, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *x = (const _Float16 *)IOSurfaceGetBaseAddress(ln->inp);

    // Lock outputs
    IOSurfaceLock(ln->out, 0, NULL);
    IOSurfaceLock(ln->k3c->ioOutputs[0], 0, NULL);
    _Float16 *norm = (_Float16 *)IOSurfaceGetBaseAddress(ln->out);
    _Float16 *rstd = (_Float16 *)IOSurfaceGetBaseAddress(ln->k3c->ioOutputs[0]);

    // Compute mean[S] and var[S] in fp32, iterating c outermost (sequential rows)
    float *mean_buf = (float *)calloc((size_t)S, sizeof(float));
    float *var_buf  = (float *)calloc((size_t)S, sizeof(float));

    // Pass 1: mean
    for (int c = 0; c < C; c++) {
        const _Float16 *xc = x + c*S;
        int s = 0;
        for (; s <= S-4; s += 4)
            vst1q_f32(mean_buf+s, vaddq_f32(vld1q_f32(mean_buf+s),
                                            vcvt_f32_f16(vld1_f16((const __fp16*)(xc+s)))));
        for (; s < S; s++) mean_buf[s] += (float)xc[s];
    }
    for (int s = 0; s < S; s++) mean_buf[s] *= inv_C;

    // Pass 2: var = mean((x-mean)^2)
    for (int c = 0; c < C; c++) {
        const _Float16 *xc = x + c*S;
        int s = 0;
        for (; s <= S-4; s += 4) {
            float32x4_t d = vsubq_f32(vcvt_f32_f16(vld1_f16((const __fp16*)(xc+s))), vld1q_f32(mean_buf+s));
            vst1q_f32(var_buf+s, vaddq_f32(vld1q_f32(var_buf+s), vmulq_f32(d, d)));
        }
        for (; s < S; s++) { float d=(float)xc[s]-mean_buf[s]; var_buf[s]+=d*d; }
    }

    // Pass 3: rstd = 1/sqrt(var/C + eps), then normalize
    for (int s = 0; s < S; s++) {
        float rs = 1.0f / sqrtf(var_buf[s] * inv_C + eps);
        rstd[s] = (_Float16)rs;
    }

    // Pass 4: norm[c,s] = (x[c,s]-mean[s])*rstd[s] — write row by row
    for (int c = 0; c < C; c++) {
        const _Float16 *xc   = x    + c*S;
        _Float16       *nc   = norm + c*S;
        int s = 0;
        for (; s <= S-4; s += 4) {
            float32x4_t rs = vcvt_f32_f16(vld1_f16((const __fp16*)(rstd+s)));
            float32x4_t mn = vld1q_f32(mean_buf+s);
            float32x4_t xv = vcvt_f32_f16(vld1_f16((const __fp16*)(xc+s)));
            vst1_f16((__fp16*)(nc+s), vcvt_f16_f32(vmulq_f32(vsubq_f32(xv, mn), rs)));
        }
        for (; s < S; s++) nc[s] = (_Float16)(((float)xc[s]-mean_buf[s])*(float)rstd[s]);
    }

    free(mean_buf); free(var_buf);
    IOSurfaceUnlock(ln->k3c->ioOutputs[0], 0, NULL);
    IOSurfaceUnlock(ln->out, 0, NULL);
    IOSurfaceUnlock(ln->inp, kIOSurfaceLockReadOnly, NULL);
}

// Threshold: 7 ANE dispatches × ~1.5ms = ~10ms overhead. CPU NEON wins above C*S > ~50K.
#define _LN_CPU_THRESHOLD (50*1024)

static void ane_ln_eval(ANELayerNorm *ln) {
    if (ln->C * ln->S > _LN_CPU_THRESHOLD) { _ane_ln_eval_cpu(ln); return; }
    ane_eval(ln->k1a);
    ane_eval(ln->k1b_2_3a);
    ane_eval(ln->k3b);
    ane_eval(ln->k3c);
    ane_eval(ln->k4);
}

static void ane_ln_read_output(ANELayerNorm *ln, _Float16 *y) {
    IOSurfaceLock(ln->out, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(ln->out), ln->C * ln->S * sizeof(_Float16));
    IOSurfaceUnlock(ln->out, kIOSurfaceLockReadOnly, NULL);
}
