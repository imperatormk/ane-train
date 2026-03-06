// ane_dw.h — ANE depthwise conv KxK, 100% on-chip, zero CPU
//
// Key insight: [1,S,C]*[1,1,C] broadcast works when large tensors are slot0.
// Decompose dw as K*K mul+add ops in one fused kernel with K*K+K*K inputs.
//
// Layout: [1,S,C] (transposed from standard [1,C,S])
//   - Input/output are [S,C] flat (row=spatial, col=channel)
//   - Weights W: [C, K*K] (channel-major)
//
// The K*K pre-shifted copies of x must be written each forward pass.
// Use ane_dw_write_shifts() to compute and write them from a [S,C] input.
//
// Usage:
//   ANEDepthwise *dw = ane_dw_compile(C, S, H, K);  // S=H*H
//   ane_dw_write_w(dw, w_fp16);          // [C*K*K] fp16, once per weight update
//   ane_dw_write_input(dw, x_sc_fp16);   // [S*C] fp16 in [S,C] layout
//   ane_dw_eval(dw);
//   ane_dw_read_output(dw, y_sc_fp16);   // [S*C] fp16 in [S,C] layout
//
#pragma once
#include "../../ane_runtime.h"
#include <arm_neon.h>

#define ANE_DW_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S, H, K, KK;
    ANEKernel *kern;
    // convenience: shift scratch buffer
    _Float16 *shift_buf;  // [S*C] temp for computing shifts
} ANEDepthwise;

static NSString *_ane_dw_mil(int C, int S, int K) {
    int KK = K * K;
    NSMutableString *m = [NSMutableString string];
    [m appendFormat:@"program(1.0)\n" ANE_DW_BI "\n{\n  func main<ios16>(\n"];
    // K[1,S,C] large FIRST (slots 0..KK-1)
    for (int k = 0; k < KK; k++)
        [m appendFormat:@"    tensor<fp16,[1,%d,%d]> K%d,\n", S, C, k];
    // W[1,1,C] small AFTER (slots KK..2*KK-1)
    for (int k = 0; k < KK; k++)
        [m appendFormat:@"    tensor<fp16,[1,1,%d]> W%d%s\n", C, k, k < KK-1 ? "," : ""];
    [m appendString:@"  ) {\n"];
    // T_k = K_k * W_k
    for (int k = 0; k < KK; k++)
        [m appendFormat:@"    tensor<fp16,[1,%d,%d]> T%d=mul(x=K%d,y=W%d)[name=tensor<string,[]>(\"T%d\")];\n",
         S, C, k, k, k, k];
    // accumulate: A0=T0+T1, A1=A0+T2, ...
    [m appendFormat:@"    tensor<fp16,[1,%d,%d]> A0=add(x=T0,y=T1)[name=tensor<string,[]>(\"A0\")];\n", S, C];
    for (int k = 2; k < KK; k++)
        [m appendFormat:@"    tensor<fp16,[1,%d,%d]> A%d=add(x=A%d,y=T%d)[name=tensor<string,[]>(\"A%d\")];\n",
         S, C, k-1, k-2, k, k-1];
    [m appendFormat:@"  } -> (A%d);\n}\n", KK - 2];
    return m;
}

static ANEDepthwise *ane_dw_compile(int C, int S, int H, int K) {
    int KK = K * K;
    size_t si = (size_t)S * C * 2;  // [1,S,C] large — NO 2048 bump
    size_t sw = (size_t)C * 2;      // [1,1,C] small — NO 2048 bump
    size_t *ins = (size_t *)malloc(2 * KK * sizeof(size_t));
    for (int k = 0; k < KK; k++) ins[k]      = si;  // large first (decreasing order)
    for (int k = 0; k < KK; k++) ins[KK + k] = sw;  // small after

    NSString *mil = _ane_dw_mil(C, S, K);
    ANEKernel *kern = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding],
                                   nil, 2*KK, ins, 1, &si);
    free(ins);
    if (!kern) {
        fprintf(stderr, "ane_dw_compile FAILED (C=%d S=%d K=%d)\n", C, S, K);
        return NULL;
    }
    ANEDepthwise *dw = (ANEDepthwise *)calloc(1, sizeof(ANEDepthwise));
    dw->C = C; dw->S = S; dw->H = H; dw->K = K; dw->KK = KK;
    dw->kern = kern;
    dw->shift_buf = (_Float16 *)malloc(S * C * sizeof(_Float16));
    return dw;
}

// Write per-position weight slice W[:,k] to slot KK+k
static void ane_dw_write_w(ANEDepthwise *dw, const _Float16 *w) {
    // w: [C, KK] — w[c*KK+k] = weight for channel c, kernel pos k
    _Float16 *wk = (_Float16 *)malloc(dw->C * sizeof(_Float16));
    for (int k = 0; k < dw->KK; k++) {
        for (int c = 0; c < dw->C; c++) wk[c] = w[c * dw->KK + k];
        IOSurfaceLock(dw->kern->ioInputs[dw->KK + k], 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(dw->kern->ioInputs[dw->KK + k]), wk, dw->C * sizeof(_Float16));
        IOSurfaceUnlock(dw->kern->ioInputs[dw->KK + k], 0, NULL);
    }
    free(wk);
}

// Compute K*K shifts from x[S,C] and write to slots 0..KK-1.
// Single-pass: read each pixel once, scatter to all K*K destination slots.
static void ane_dw_write_input(ANEDepthwise *dw, const _Float16 *x_sc) {
    int H = dw->H, C = dw->C, K = dw->K, KK = dw->KK, pad = K / 2;
    size_t SC = (size_t)dw->S * C;

    // Lock all slots and collect base pointers upfront
    _Float16 *dst[KK];
    for (int k = 0; k < KK; k++) {
        IOSurfaceLock(dw->kern->ioInputs[k], 0, NULL);
        dst[k] = (_Float16 *)IOSurfaceGetBaseAddress(dw->kern->ioInputs[k]);
        memset(dst[k], 0, SC * sizeof(_Float16));
    }

    // Single pass over input: for each source pixel (ih,iw), load src into
    // NEON registers once, then store to all KK destination slots.
    // C must be multiple of 8 (guaranteed by ANE matmul Ci constraint ≥32).
    for (int ih = 0; ih < H; ih++) for (int iw = 0; iw < H; iw++) {
        const _Float16 *src = x_sc + (ih*H + iw) * C;
        // Preload src row into NEON registers
        float16x8_t regs[C / 8];
        for (int r = 0; r < C/8; r++)
            regs[r] = vld1q_f16((const __fp16 *)(src + r*8));
        // Scatter to each valid dst slot
        for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++) {
            int oh = ih - (ky - pad);
            int ow = iw - (kx - pad);
            if (oh < 0 || oh >= H || ow < 0 || ow >= H) continue;
            __fp16 *d = (__fp16 *)(dst[ky*K+kx] + (oh*H + ow)*C);
            for (int r = 0; r < C/8; r++)
                vst1q_f16(d + r*8, regs[r]);
        }
    }

    for (int k = 0; k < KK; k++)
        IOSurfaceUnlock(dw->kern->ioInputs[k], 0, NULL);
}

static void ane_dw_eval(ANEDepthwise *dw) { ane_eval(dw->kern); }

static void ane_dw_read_output(ANEDepthwise *dw, _Float16 *y_sc) {
    IOSurfaceLock(dw->kern->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(y_sc, IOSurfaceGetBaseAddress(dw->kern->ioOutputs[0]), dw->S * dw->C * sizeof(_Float16));
    IOSurfaceUnlock(dw->kern->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
}

// Write shifted inputs directly from [C,S] layout (no separate transpose needed).
// Step 1: blocked transpose x_cs[C,S] → shift_buf[S,C] in cacheable RAM
// Step 2: scatter from shift_buf[S,C] → 49 dw IOSurfaces (existing ane_dw_write_input)
static void ane_dw_write_input_cs(ANEDepthwise *dw, const _Float16 *x_cs) {
    int C = dw->C, S = dw->S;
    // Blocked transpose [C,S] → [S,C] in cacheable shift_buf
    for (int c = 0; c < C; c += 32) for (int s = 0; s < S; s += 32) {
        int ce = c+32 < C ? c+32 : C, se = s+32 < S ? s+32 : S;
        for (int ci = c; ci < ce; ci++) for (int si = s; si < se; si++)
            dw->shift_buf[si*C + ci] = x_cs[ci*S + si];
    }
    // Now scatter from cacheable RAM to IOSurfaces
    ane_dw_write_input(dw, dw->shift_buf);
}

// Read dw output [S,C] and transpose to [C,S] in dst buffer.
// Two-step: sequential memcpy from IOSurface, then blocked transpose in cacheable RAM.
static void ane_dw_read_output_cs(ANEDepthwise *dw, _Float16 *y_cs) {
    int C = dw->C, S = dw->S;
    // Step 1: bulk copy from IOSurface to scratch (sequential, fast)
    IOSurfaceLock(dw->kern->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(dw->shift_buf, IOSurfaceGetBaseAddress(dw->kern->ioOutputs[0]), (size_t)S*C*2);
    IOSurfaceUnlock(dw->kern->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    // Step 2: blocked transpose [S,C] → [C,S] in cacheable RAM
    const _Float16 *src = dw->shift_buf;
    for (int s = 0; s < S; s += 32) for (int c = 0; c < C; c += 32) {
        int se = s+32 < S ? s+32 : S, ce = c+32 < C ? c+32 : C;
        for (int si = s; si < se; si++) {
            const _Float16 *row = src + si*C + c;
            for (int ci = c; ci < ce; ci++)
                y_cs[ci*S + si] = row[ci - c];
        }
    }
}

// Expose input/output surfaces for chaining
static IOSurfaceRef ane_dw_input_surf(ANEDepthwise *dw, int k) { return dw->kern->ioInputs[k]; }
static IOSurfaceRef ane_dw_output_surf(ANEDepthwise *dw)        { return dw->kern->ioOutputs[0]; }
