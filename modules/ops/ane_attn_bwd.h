// ane_attn_bwd.h — Self-attention backward (ANE + CPU for softmax bwd)
//
// Forward recap:
//   Q=Wq@x, K=Wk@x, V=Wv@x
//   scores = Q^T@K / sqrt(C)  [S,S]
//   attn   = softmax(scores)   [S,S]
//   out    = V@attn            [C,S]
//   proj   = Wo@out            [C,S]
//   y      = proj + x          (residual)
//
// Backward:
//   dy[C,S] arrives (= gradient of loss w.r.t. y)
//   1. residual: dx_res = dy (pass-through)
//   2. out-proj bwd: d_out[C,S] = Wo^T @ dy,  dWo = dy @ out^T
//   3. V@attn bwd:
//       dV[C,S]   = d_out @ attn^T
//       d_attn[S,S] = V^T @ d_out
//   4. softmax bwd (CPU, [S,S]):
//       d_scores[i,j] = attn[i,j] * (d_attn[i,j] - sum_k(d_attn[i,k]*attn[i,k]))
//       d_scores /= sqrt(C)
//   5. scores = Q^T @ K bwd:
//       dQ[C,S] = K @ d_scores^T
//       dK[C,S] = Q @ d_scores
//   6. QKV proj bwd:
//       dWq = dQ @ x^T,  dWk = dK @ x^T,  dWv = dV @ x^T
//       dx_qkv = Wq^T@dQ + Wk^T@dK + Wv^T@dV
//   7. dx_total = dx_res + dx_qkv
//
// Saved activations needed:
//   Q[C,S], K[C,S], V[C,S], out[C,S], attn[S,S], x[C,S]
//
// ANE kernels for the heavy matmuls, CPU for softmax bwd (S×S element-wise).
//
#pragma once
#include "ane_attn.h"
#include "ane_matmul_bwd.h"
#include <arm_neon.h>
#include <math.h>

#define ANE_ATTN_BWD_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

// Softmax backward: d_scores[S,S] = attn*(d_attn - rowsum(d_attn*attn))
// Both [S,S] in row-major. Done CPU because it's just S×S element ops.
static void _attn_softmax_bwd(const _Float16 *attn, const _Float16 *d_attn,
                               _Float16 *d_scores, int S, float inv_sqrt_C) {
    for (int i = 0; i < S; i++) {
        // dot product: sum_j d_attn[i,j] * attn[i,j]
        float dot = 0.0f;
        for (int j = 0; j < S; j++)
            dot += (float)d_attn[i*S+j] * (float)attn[i*S+j];
        // d_scores[i,j] = attn[i,j] * (d_attn[i,j] - dot) / sqrt(C)
        for (int j = 0; j < S; j++)
            d_scores[i*S+j] = (_Float16)(
                (float)attn[i*S+j] * ((float)d_attn[i*S+j] - dot) * inv_sqrt_C);
    }
}

// Simple ANE kernel: A[Ca,S] @ B^T[Cb,S] → C[Ca,Cb]  (dW-style)
// Used for dWq, dWk, dWv, dWo.
// Reuses mil_gen_dW(Ci=S, Co=Ca, ... wait — dW is dy@x^T: inputs dy[Co,S] and x[Ci,S]
// Here we want: dQ[C,S] @ x^T[S,C] → dWq[C,C]. Same shape as mil_gen_dW(Ci=C, Co=C, S).
// So we reuse ANEMatmulBwd's k_dw kernel.

typedef struct {
    int C, S;
    // Saved fwd activations (allocated here, filled after each fwd pass)
    _Float16 *Q, *K, *V, *out_proj, *attn_weights, *x_saved; // [C,S], [C,S], [C,S], [C,S], [S,S], [C,S]

    // Bwd kernels (all matmul-based)
    ANEMatmulBwd *Wo_bwd;   // dy[C,S] → d_out[C,S], dWo[C,C]; W=Wo, x=out_proj
    // V@attn bwd: d_out[C,S] @ attn^T[S,S] → dV[C,S];  V^T[S,C] @ d_out[C,S] → d_attn_vs[S,S]
    // We need two separate matmuls here:
    //   dV   = d_out[C,S] @ attn[S,S]^T  → matmul(d_out, attn, ty=true) = [C,S]
    //   d_attn = V^T[S,C] @ d_out[C,S]   → matmul(V, d_out, tx=true) = [S,S]
    // These are not standard matmul_bwd shape — compile custom kernels.
    ANEKernel *k_dV;       // (d_out[C,S], attn[S,S]) → dV[C,S]
    ANEKernel *k_dattn;    // (V[C,S], d_out[C,S]) → d_attn[S,S]
    // scores bwd:
    //   dQ[C,S] = K[C,S] @ d_scores^T[S,S] → matmul(K, d_scores, ty=true)
    //   dK[C,S] = Q[C,S] @ d_scores[S,S]
    ANEKernel *k_dQ;       // (K[C,S], d_scores[S,S]) → dQ[C,S]
    ANEKernel *k_dK;       // (Q[C,S], d_scores[S,S]) → dK[C,S]
    // QKV proj bwd: dWq = dQ@x^T, etc. — use ANEMatmulBwd dW kernels
    ANEMatmulBwd *Wq_bwd;  // x[C,S], dQ[C,S] → dWq[C,C], dxq[C,S]
    ANEMatmulBwd *Wk_bwd;  // x[C,S], dK[C,S] → dWk[C,C], dxk[C,S]
    ANEMatmulBwd *Wv_bwd;  // x[C,S], dV[C,S] → dWv[C,C], dxv[C,S]

    // Scratch
    _Float16 *d_out;       // [C,S]
    _Float16 *dV;          // [C,S]
    _Float16 *d_attn_raw;  // [S,S]
    _Float16 *d_scores;    // [S,S]
    _Float16 *dQ;          // [C,S]
    _Float16 *dK;          // [C,S]
    _Float16 *dx_q, *dx_k, *dx_v; // [C,S] each
} ANEAttnBwd;

// Custom kernel: A[1,Ca,Sb] @ B[1,Sb,Sc]^T → C[1,Ca,Sc]  (transpose_y=true)
static ANEKernel *_attn_bwd_compile_matmul_ty(int Ca, int Sb, int Sc) {
    // slot rule: smaller first. A=[Ca,Sb], B=[Sb,Sc] stored as [Sc,Sb] after transpose.
    // actual sizes: sA=Ca*Sb*2, sB=Sb*Sc*2 → put smaller first
    size_t sA = (size_t)Ca * Sb * 2;  if (sA < 2048) sA = 2048;
    size_t sB = (size_t)Sb * Sc * 2;  if (sB < 2048) sB = 2048;
    size_t sC = (size_t)Ca * Sc * 2;  if (sC < 2048) sC = 2048;
    // MIL: matmul(A[Ca,Sb], B[Sb,Sc], tx=ff, ty=tt) → [Ca,Sc]
    // We declare B as [1,Sc,Sb] in MIL so transpose_y=true gives [Sb,Sc] effectively.
    // Actually easier: declare B[1,Sb,Sc] and transpose_y=true → matmul of A[Ca,Sb] @ B^T[Sc,Sb] = [Ca,Sc]
    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BWD_BI "\n{\n"
        "  func main<ios16>(tensor<fp16,[1,%d,%d]> A, tensor<fp16,[1,%d,%d]> B) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<fp16,[1,%d,%d]> C=matmul(transpose_x=ff,transpose_y=tt,x=A,y=B)"
        "[name=tensor<string,[]>(\"C\")];\n"
        "  } -> (C);\n}\n",
        Ca, Sb,   // A shape
        Sb, Sc,   // B shape (we transpose it)
        Ca, Sc];  // output
    size_t ins[2] = {sA < sB ? sA : sB,  sA < sB ? sB : sA};
    // Always put smaller first
    if (sA > sB) {
        // swap declared order in MIL — just put B first if B is smaller
        // Simpler: just use a consistent order and accept possible slot violation warning
        // For our cases: Ca=C,Sb=S,Sc=S → sA=C*S, sB=S*S → sA<sB for C<S ✓
    }
    size_t ins2[2] = {sA, sB};
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins2, 1, &sC);
    return k;
}

// Custom kernel: A[1,Ca,Sb]^T @ B[1,Ca,Sc] → C[1,Sb,Sc]  (transpose_x=true)
static ANEKernel *_attn_bwd_compile_matmul_tx(int Ca, int Sb, int Sc) {
    size_t sA = (size_t)Ca * Sb * 2;  if (sA < 2048) sA = 2048;
    size_t sB = (size_t)Ca * Sc * 2;  if (sB < 2048) sB = 2048;
    size_t sC = (size_t)Sb * Sc * 2;  if (sC < 2048) sC = 2048;
    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BWD_BI "\n{\n"
        "  func main<ios16>(tensor<fp16,[1,%d,%d]> A, tensor<fp16,[1,%d,%d]> B) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<fp16,[1,%d,%d]> C=matmul(transpose_x=tt,transpose_y=ff,x=A,y=B)"
        "[name=tensor<string,[]>(\"C\")];\n"
        "  } -> (C);\n}\n",
        Ca, Sb,   // A [Ca,Sb] → transposed to [Sb,Ca]
        Ca, Sc,   // B [Ca,Sc]
        Sb, Sc];  // output [Sb,Sc]
    size_t ins[2] = {sA, sB};
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins, 1, &sC);
    return k;
}

static ANEAttnBwd *ane_attn_bwd_compile(ANEAttn *fwd) {
    int C = fwd->C, S = fwd->S;
    ANEAttnBwd *bwd = (ANEAttnBwd *)calloc(1, sizeof(ANEAttnBwd));
    bwd->C = C; bwd->S = S;

    // Wo bwd: Ci=C, Co=C, S=S  (dy[C,S] → d_out[C,S], dWo[C,C])
    bwd->Wo_bwd = ane_matmul_bwd_compile(C, C, S);
    if (!bwd->Wo_bwd) { fprintf(stderr,"ane_attn_bwd Wo_bwd FAIL\n"); return NULL; }
    // Wire Wo weight — fwd->kout slot0 is Wo
    ane_matmul_bwd_rewire_w(bwd->Wo_bwd, fwd->kout->ioInputs[0]);

    // V@attn bwd:
    // k_dV: d_out[C,S] @ attn[S,S]^T → dV[C,S]   → matmul_ty(Ca=C, Sb=S, Sc=S)
    bwd->k_dV    = _attn_bwd_compile_matmul_ty(C, S, S);
    // k_dattn: V[C,S]^T @ d_out[C,S] → d_attn[S,S] → matmul_tx(Ca=C, Sb=S, Sc=S)
    bwd->k_dattn = _attn_bwd_compile_matmul_tx(C, S, S);

    // scores bwd:
    // k_dQ: K[C,S] @ d_scores[S,S]^T → dQ[C,S]  → matmul_ty(Ca=C, Sb=S, Sc=S)
    bwd->k_dQ = _attn_bwd_compile_matmul_ty(C, S, S);
    // k_dK: Q[C,S] @ d_scores[S,S] → dK[C,S]    → standard matmul(Ca=C, Sb=S, Sc=S)
    // We want C[C,S] = A[C,S] @ B[S,S], tx=ff, ty=ff
    {
        size_t sA = (size_t)C * S * 2; if (sA < 2048) sA = 2048;
        size_t sB = (size_t)S * S * 2; if (sB < 2048) sB = 2048;
        size_t sC2 = (size_t)C * S * 2; if (sC2 < 2048) sC2 = 2048;
        NSString *mil = [NSString stringWithFormat:
            @"program(1.0)\n" ANE_ATTN_BWD_BI "\n{\n"
            "  func main<ios16>(tensor<fp16,[1,%d,%d]> A, tensor<fp16,[1,%d,%d]> B) {\n"
            "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
            "    tensor<fp16,[1,%d,%d]> C=matmul(transpose_x=ff,transpose_y=ff,x=A,y=B)"
            "[name=tensor<string,[]>(\"C\")];\n"
            "  } -> (C);\n}\n",
            C, S,  S, S,  C, S];
        size_t ins[2] = {sA, sB};
        bwd->k_dK = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins, 1, &sC2);
    }

    if (!bwd->k_dV||!bwd->k_dattn||!bwd->k_dQ||!bwd->k_dK) {
        fprintf(stderr,"ane_attn_bwd V/attn/scores FAIL\n"); return NULL;
    }

    // QKV proj bwd: Ci=C, Co=C, S=S for all three
    bwd->Wq_bwd = ane_matmul_bwd_compile(C, C, S);
    bwd->Wk_bwd = ane_matmul_bwd_compile(C, C, S);
    bwd->Wv_bwd = ane_matmul_bwd_compile(C, C, S);
    if (!bwd->Wq_bwd||!bwd->Wk_bwd||!bwd->Wv_bwd) {
        fprintf(stderr,"ane_attn_bwd QKV FAIL\n"); return NULL;
    }
    // Wire weights: kq/kk/kv slot0 = Wq/Wk/Wv
    ane_matmul_bwd_rewire_w(bwd->Wq_bwd, fwd->kq->ioInputs[0]);
    ane_matmul_bwd_rewire_w(bwd->Wk_bwd, fwd->kk->ioInputs[0]);
    ane_matmul_bwd_rewire_w(bwd->Wv_bwd, fwd->kv->ioInputs[0]);

    // Saved activations
    bwd->Q            = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->K            = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->V            = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->out_proj     = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->attn_weights = (_Float16 *)malloc((size_t)S * S * 2);
    bwd->x_saved      = (_Float16 *)malloc((size_t)C * S * 2);

    // Scratch
    bwd->d_out      = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->dV         = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->d_attn_raw = (_Float16 *)malloc((size_t)S * S * 2);
    bwd->d_scores   = (_Float16 *)malloc((size_t)S * S * 2);
    bwd->dQ         = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->dK         = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->dx_q = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->dx_k = (_Float16 *)malloc((size_t)C * S * 2);
    bwd->dx_v = (_Float16 *)malloc((size_t)C * S * 2);

    return bwd;
}

// Save forward activations. Call immediately after ane_attn_eval().
static void ane_attn_bwd_save_fwd(ANEAttnBwd *bwd, ANEAttn *fwd) {
    int C = bwd->C, S = bwd->S;
    // Q = kq output, K = kk output, V = kv output
    IOSurfaceLock(fwd->kq->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->Q, IOSurfaceGetBaseAddress(fwd->kq->ioOutputs[0]), C*S*2);
    IOSurfaceUnlock(fwd->kq->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(fwd->kk->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->K, IOSurfaceGetBaseAddress(fwd->kk->ioOutputs[0]), C*S*2);
    IOSurfaceUnlock(fwd->kk->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(fwd->kv->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->V, IOSurfaceGetBaseAddress(fwd->kv->ioOutputs[0]), C*S*2);
    IOSurfaceUnlock(fwd->kv->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    // attn = kdiv output
    IOSurfaceLock(fwd->kdiv->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->attn_weights, IOSurfaceGetBaseAddress(fwd->kdiv->ioOutputs[0]), S*S*2);
    IOSurfaceUnlock(fwd->kdiv->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    // out = kvattn output
    IOSurfaceLock(fwd->kvattn->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->out_proj, IOSurfaceGetBaseAddress(fwd->kvattn->ioOutputs[0]), C*S*2);
    IOSurfaceUnlock(fwd->kvattn->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    // x = fwd input
    IOSurfaceLock(fwd->x_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->x_saved, IOSurfaceGetBaseAddress(fwd->x_surf), C*S*2);
    IOSurfaceUnlock(fwd->x_surf, kIOSurfaceLockReadOnly, NULL);
}

// Helper: write [Ca,Sb] to slot0 and [Sb,Sc] to slot1 of a kernel
static void _attn_bwd_write2(ANEKernel *k, const _Float16 *a, size_t sa,
                              const _Float16 *b, size_t sb) {
    IOSurfaceLock(k->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[0]), a, sa);
    IOSurfaceUnlock(k->ioInputs[0], 0, NULL);
    IOSurfaceLock(k->ioInputs[1], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[1]), b, sb);
    IOSurfaceUnlock(k->ioInputs[1], 0, NULL);
}

// Run full attention backward. dy[C,S], dx[C,S] output.
// dWq/dWk/dWv/dWo written to provided buffers (caller uses for Adam).
static void ane_attn_bwd_eval(ANEAttnBwd *bwd, ANEAttn *fwd,
                               const _Float16 *dy, _Float16 *dx,
                               _Float16 *dWq, _Float16 *dWk,
                               _Float16 *dWv, _Float16 *dWo) {
    int C = bwd->C, S = bwd->S;

    // 1. Residual pass-through: dx gets dy added at the end

    // 2. Out-proj bwd: Wo^T @ dy → d_out, dy @ out^T → dWo
    ane_matmul_bwd_write_dy(bwd->Wo_bwd, dy);
    ane_matmul_bwd_rewire_x(bwd->Wo_bwd, fwd->kvattn->ioOutputs[0]); // out on ANE surface
    ane_matmul_bwd_eval(bwd->Wo_bwd);
    ane_matmul_bwd_read_dx(bwd->Wo_bwd, bwd->d_out);
    if (dWo) ane_matmul_bwd_read_dw(bwd->Wo_bwd, dWo);

    // 3a. dV = d_out[C,S] @ attn^T[S,S]
    _attn_bwd_write2(bwd->k_dV,
        bwd->d_out,       (size_t)C * S * 2,
        bwd->attn_weights,(size_t)S * S * 2);
    ane_eval(bwd->k_dV);
    IOSurfaceLock(bwd->k_dV->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->dV, IOSurfaceGetBaseAddress(bwd->k_dV->ioOutputs[0]), C*S*2);
    IOSurfaceUnlock(bwd->k_dV->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);

    // 3b. d_attn = V^T[S,C] @ d_out[C,S] → [S,S]
    _attn_bwd_write2(bwd->k_dattn,
        bwd->V,     (size_t)C * S * 2,
        bwd->d_out, (size_t)C * S * 2);
    ane_eval(bwd->k_dattn);
    IOSurfaceLock(bwd->k_dattn->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->d_attn_raw, IOSurfaceGetBaseAddress(bwd->k_dattn->ioOutputs[0]), S*S*2);
    IOSurfaceUnlock(bwd->k_dattn->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);

    // 4. Softmax bwd (CPU)
    _attn_softmax_bwd(bwd->attn_weights, bwd->d_attn_raw, bwd->d_scores, S, 1.0f/sqrtf((float)C));

    // 5a. dQ = K @ d_scores^T[S,S]
    _attn_bwd_write2(bwd->k_dQ,
        bwd->K,        (size_t)C * S * 2,
        bwd->d_scores, (size_t)S * S * 2);
    ane_eval(bwd->k_dQ);
    IOSurfaceLock(bwd->k_dQ->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->dQ, IOSurfaceGetBaseAddress(bwd->k_dQ->ioOutputs[0]), C*S*2);
    IOSurfaceUnlock(bwd->k_dQ->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);

    // 5b. dK = Q @ d_scores[S,S]
    _attn_bwd_write2(bwd->k_dK,
        bwd->Q,        (size_t)C * S * 2,
        bwd->d_scores, (size_t)S * S * 2);
    ane_eval(bwd->k_dK);
    IOSurfaceLock(bwd->k_dK->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    memcpy(bwd->dK, IOSurfaceGetBaseAddress(bwd->k_dK->ioOutputs[0]), C*S*2);
    IOSurfaceUnlock(bwd->k_dK->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);

    // 6. QKV proj bwd: dW and dx for each
    // Wq: dy=dQ, x=x_saved
    ane_matmul_bwd_write_dy(bwd->Wq_bwd, bwd->dQ);
    ane_matmul_bwd_write_x(bwd->Wq_bwd, bwd->x_saved);
    ane_matmul_bwd_eval(bwd->Wq_bwd);
    ane_matmul_bwd_read_dx(bwd->Wq_bwd, bwd->dx_q);
    if (dWq) ane_matmul_bwd_read_dw(bwd->Wq_bwd, dWq);

    ane_matmul_bwd_write_dy(bwd->Wk_bwd, bwd->dK);
    ane_matmul_bwd_write_x(bwd->Wk_bwd, bwd->x_saved);
    ane_matmul_bwd_eval(bwd->Wk_bwd);
    ane_matmul_bwd_read_dx(bwd->Wk_bwd, bwd->dx_k);
    if (dWk) ane_matmul_bwd_read_dw(bwd->Wk_bwd, dWk);

    ane_matmul_bwd_write_dy(bwd->Wv_bwd, bwd->dV);
    ane_matmul_bwd_write_x(bwd->Wv_bwd, bwd->x_saved);
    ane_matmul_bwd_eval(bwd->Wv_bwd);
    ane_matmul_bwd_read_dx(bwd->Wv_bwd, bwd->dx_v);
    if (dWv) ane_matmul_bwd_read_dw(bwd->Wv_bwd, dWv);

    // 7. Accumulate dx = residual(dy) + dx_q + dx_k + dx_v
    for (int i = 0; i < C*S; i++)
        dx[i] = (_Float16)((float)dy[i]
                         + (float)bwd->dx_q[i]
                         + (float)bwd->dx_k[i]
                         + (float)bwd->dx_v[i]);
}
