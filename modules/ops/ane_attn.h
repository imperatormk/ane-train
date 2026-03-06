// ane_attn.h — ANE single-head self-attention: Q,K,V projections + softmax(QK^T/√d)V
//
// Layout: [1, C, S] throughout (C = head dim, S = sequence length = H*W)
//
// Forward:
//   Q = Wq[C,C] @ x[C,S]          → Q[C,S]
//   K = Wk[C,C] @ x[C,S]          → K[C,S]
//   V = Wv[C,C] @ x[C,S]          → V[C,S]
//   scores[S,S] = Q^T[S,C] @ K[C,S] / sqrt(C)   (transpose_x=true on Q)
//   exp_sc[S,S] = exp(scores)
//   rowsum[1,S] = ones[1,S] @ exp_sc^T[S,S]      (transpose_y=true on exp_sc)
//   attn[S,S]   = exp_sc / rowsum                 (broadcast [1,S] over rows)
//   out[C,S]    = V[C,S] @ attn[S,S]
//   y[C,S]      = Wo[C,C] @ out[C,S] + x[C,S]    (residual)
//
// Constraints:
//   - C must be multiple of 32 (ANE matmul Ci constraint)
//   - S must satisfy C ≤ S (slot rule: W[C,C] ≤ x[C,S])
//   - S*S must fit in IOSurface (S=1024: 2MB — fine)
//
// Kernels (compiled once):
//   kq, kk, kv: 3× matmul [C,C]@[C,S] → [C,S]   (QKV projections)
//   kqk:        matmul(Q^T[S,C] @ K[C,S]) → [S,S] (scores, transpose_x=true)
//   kexp:       exp([S,S]) → [S,S]
//   krowsum:    matmul(ones[1,S] @ exp^T[S,S]) → [1,S]  (transpose_y=true)
//   kdiv:       div(exp[S,S], rowsum[1,S]) → attn[S,S]  (broadcast)
//   kvattn:     matmul(V[C,S] @ attn[S,S]) → out[C,S]
//   kout:       matmul(Wo[C,C] @ out[C,S]) → [C,S]
//   kadd:       add(kout_y, x) → y[C,S]                 (residual)
//
// Usage:
//   ANEAttn *attn = ane_attn_compile(C, S);
//   ane_attn_set_weights(attn, Wq, Wk, Wv, Wo);  // each [C*C] fp16
//   ane_attn_write_x(attn, x_fp16);               // [C*S]
//   ane_attn_eval(attn);
//   ane_attn_read_y(attn, y_fp16);                // [C*S]
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_ATTN_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]\n"

typedef struct {
    int C, S;
    // kernels
    ANEKernel *kq, *kk, *kv;   // QKV projections
    ANEKernel *kqk;              // scores = Q^T @ K
    ANEKernel *kexp;             // exp(scores)
    ANEKernel *krowsum;          // rowsum via ones @ exp^T
    ANEKernel *kdiv;             // attn = exp / rowsum
    ANEKernel *kvattn;           // out = V @ attn
    ANEKernel *kout;             // projected = Wo @ out
    ANEKernel *kadd;             // y = projected + x (residual)
    // IOSurfaces for input/output
    IOSurfaceRef x_surf;         // input x[C,S]
    IOSurfaceRef y_surf;         // output y[C,S]
    // ones vector for rowsum
    IOSurfaceRef ones_surf;      // [1,S] all-ones
} ANEAttn;

// --- MIL generators ---

static NSString *_attn_mil_proj(int C, int S) {
    // matmul: W[1,C,C] @ x[1,C,S] -> y[1,C,S]
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> W, tensor<fp16,[1,%d,%d]> X) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,%d,%d]> Y=matmul(transpose_x=ff,transpose_y=ff,x=W,y=X)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        C,C, C,S, C,S];
}

static NSString *_attn_mil_scores(int C, int S) {
    // scores[S,S] = Q^T[S,C] @ K[C,S]  (transpose_x=true on Q)
    // Both inputs [C,S], equal size. slot0=Q, slot1=K (arbitrary when equal).
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> Q, tensor<fp16,[1,%d,%d]> K) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<fp16,[]> SC=const()[name=tensor<string,[]>(\"SC\"),val=tensor<fp16,[]>(%f)];\n"
        "    tensor<fp16,[1,%d,%d]> RAW=matmul(transpose_x=tt,transpose_y=ff,x=Q,y=K)[name=tensor<string,[]>(\"RAW\")];\n"
        "    tensor<fp16,[1,%d,%d]> SCORES=mul(x=RAW,y=SC)[name=tensor<string,[]>(\"SCORES\")];\n"
        "  } -> (SCORES);\n}\n",
        C,S, C,S, 1.0f/sqrtf((float)C), S,S, S,S];
}

static NSString *_attn_mil_exp(int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> X) {\n"
        "    tensor<fp16,[1,%d,%d]> Y=exp(x=X)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        S,S, S,S];
}

static NSString *_attn_mil_rowsum(int S) {
    // rowsum[1,S] = ones[1,1,S] @ exp_sc^T[1,S,S]
    // ones[1,S]=slot0(small), exp_sc[S,S]=slot1(large)
    // matmul(ones, exp_sc, transpose_y=true) -> [1,1,S] rowsums
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,1,%d]> ONES, tensor<fp16,[1,%d,%d]> ESC) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<fp16,[1,1,%d]> RS=matmul(transpose_x=ff,transpose_y=tt,x=ONES,y=ESC)[name=tensor<string,[]>(\"RS\")];\n"
        "  } -> (RS);\n}\n",
        S, S,S, S];
}

static NSString *_attn_mil_div(int S) {
    // attn[S,S] = exp_sc[S,S] / rowsum[1,S]  (broadcast rowsum over rows)
    // exp_sc[S,S]=slot0(large? no — rowsum[1,S] is smaller)
    // slot0=rowsum[1,S](small), slot1=exp_sc[S,S](large)
    // real_div(x=exp_sc, y=rowsum) with broadcast
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,1,%d]> RS, tensor<fp16,[1,%d,%d]> ESC) {\n"
        "    tensor<fp16,[1,%d,%d]> ATTN=real_div(x=ESC,y=RS)[name=tensor<string,[]>(\"ATTN\")];\n"
        "  } -> (ATTN);\n}\n",
        S, S,S, S,S];
}

static NSString *_attn_mil_vattn(int C, int S) {
    // out[C,S] = V[C,S] @ attn[S,S]
    // slot0=V[C,S](small when C<S), slot1=attn[S,S](large)
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> V, tensor<fp16,[1,%d,%d]> ATTN) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,%d,%d]> OUT=matmul(transpose_x=ff,transpose_y=ff,x=V,y=ATTN)[name=tensor<string,[]>(\"OUT\")];\n"
        "  } -> (OUT);\n}\n",
        C,S, S,S, C,S];
}

static NSString *_attn_mil_add(int C, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_ATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> A, tensor<fp16,[1,%d,%d]> B) {\n"
        "    tensor<fp16,[1,%d,%d]> Y=add(x=A,y=B)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        C,S, C,S, C,S];
}

static ANEAttn *ane_attn_compile(int C, int S) {
    if (C % 32 != 0) {
        fprintf(stderr, "ane_attn_compile: C=%d must be multiple of 32\n", C);
        return NULL;
    }
    if (C > S) {
        fprintf(stderr, "ane_attn_compile: C=%d must be <= S=%d (slot rule)\n", C, S);
        return NULL;
    }

    size_t sCC = (size_t)C*C*2;   if (sCC < 2048) sCC = 2048;
    size_t sCS = (size_t)C*S*2;   if (sCS < 2048) sCS = 2048;
    size_t sSS = (size_t)S*S*2;   if (sSS < 2048) sSS = 2048;
    size_t sS1 = (size_t)S*1*2;   if (sS1 < 2048) sS1 = 2048;

    // QKV projection: W[C,C](slot0) @ x[C,S](slot1)
    size_t ins_proj[2] = {sCC, sCS};
    // scores: Q[C,S](slot0) @ K[C,S](slot1) — equal size
    size_t ins_qk[2] = {sCS, sCS};
    // exp: [S,S] -> [S,S]
    // rowsum: ones[1,S](slot0) @ exp[S,S](slot1)
    size_t ins_rs[2] = {sS1, sSS};
    // div: rowsum[1,S](slot0) @ exp[S,S](slot1)
    size_t ins_div[2] = {sS1, sSS};
    // V@attn: V[C,S](slot0) @ attn[S,S](slot1)
    size_t ins_va[2] = {sCS, sSS};
    // out proj: W[C,C](slot0) @ out[C,S](slot1)
    // add: [C,S](slot0) + [C,S](slot1)
    size_t ins_add[2] = {sCS, sCS};

#define COMPILE(name, mil, ni, ins, no, outs) \
    ANEKernel *name = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, ni, ins, no, outs); \
    if (!name) { fprintf(stderr, "ane_attn_compile: " #name " FAILED (C=%d S=%d)\n", C, S); return NULL; }

    COMPILE(kq,      _attn_mil_proj(C,S),    2, ins_proj, 1, &sCS);
    COMPILE(kk,      _attn_mil_proj(C,S),    2, ins_proj, 1, &sCS);
    COMPILE(kv,      _attn_mil_proj(C,S),    2, ins_proj, 1, &sCS);
    COMPILE(kqk,     _attn_mil_scores(C,S),  2, ins_qk,   1, &sSS);
    COMPILE(kexp,    _attn_mil_exp(S),       1, &sSS,     1, &sSS);
    COMPILE(krowsum, _attn_mil_rowsum(S),    2, ins_rs,   1, &sS1);
    COMPILE(kdiv,    _attn_mil_div(S),       2, ins_div,  1, &sSS);
    COMPILE(kvattn,  _attn_mil_vattn(C,S),   2, ins_va,   1, &sCS);
    COMPILE(kout,    _attn_mil_proj(C,S),    2, ins_proj, 1, &sCS);
    COMPILE(kadd,    _attn_mil_add(C,S),     2, ins_add,  1, &sCS);
#undef COMPILE

    ANEAttn *a = (ANEAttn *)calloc(1, sizeof(ANEAttn));
    a->C = C; a->S = S;
    a->kq = kq; a->kk = kk; a->kv = kv;
    a->kqk = kqk; a->kexp = kexp; a->krowsum = krowsum;
    a->kdiv = kdiv; a->kvattn = kvattn; a->kout = kout; a->kadd = kadd;

    // Wire surfaces: chain outputs into next kernel inputs
    // kq/kk/kv: input x_surf (slot1) will be set to same x_surf
    // scores(kqk): slot0=Q=kq->out, slot1=K=kk->out
    {
        IOSurfaceRef ins[2] = {kq->ioOutputs[0], kk->ioOutputs[0]};
        ane_rewire(kqk, ins, NULL);
    }
    // exp(kexp): slot0=scores=kqk->out
    {
        IOSurfaceRef ins[1] = {kqk->ioOutputs[0]};
        ane_rewire(kexp, ins, NULL);
    }
    // rowsum(krowsum): slot0=ones(to be filled), slot1=exp_sc=kexp->out
    {
        IOSurfaceRef ins[2] = {krowsum->ioInputs[0], kexp->ioOutputs[0]};
        ane_rewire(krowsum, ins, NULL);
    }
    // div(kdiv): slot0=rowsum=krowsum->out, slot1=exp_sc=kexp->out
    {
        IOSurfaceRef ins[2] = {krowsum->ioOutputs[0], kexp->ioOutputs[0]};
        ane_rewire(kdiv, ins, NULL);
    }
    // vattn(kvattn): slot0=V=kv->out, slot1=attn=kdiv->out
    {
        IOSurfaceRef ins[2] = {kv->ioOutputs[0], kdiv->ioOutputs[0]};
        ane_rewire(kvattn, ins, NULL);
    }
    // out proj(kout): slot0=Wo (weight, to be set), slot1=vattn_out=kvattn->out
    {
        IOSurfaceRef ins[2] = {kout->ioInputs[0], kvattn->ioOutputs[0]};
        ane_rewire(kout, ins, NULL);
    }
    // add(kadd): slot0=out_proj=kout->out, slot1=x (original input, to be set)
    {
        IOSurfaceRef ins[2] = {kout->ioOutputs[0], NULL};
        ane_rewire(kadd, ins, NULL);
    }

    // Fill ones vector (doesn't change)
    a->ones_surf = krowsum->ioInputs[0];
    IOSurfaceLock(a->ones_surf, 0, NULL);
    _Float16 *ones_ptr = (_Float16 *)IOSurfaceGetBaseAddress(a->ones_surf);
    for (int i = 0; i < S; i++) ones_ptr[i] = (_Float16)1.0f;
    IOSurfaceUnlock(a->ones_surf, 0, NULL);

    // x_surf = shared input: goes to kq slot1, kk slot1, kv slot1, kadd slot1
    a->x_surf = kq->ioInputs[1];  // will write here; rewire others to same surface
    {
        IOSurfaceRef ins[2] = {NULL, kq->ioInputs[1]};
        ane_rewire(kk, ins, NULL);
    }
    {
        IOSurfaceRef ins[2] = {NULL, kq->ioInputs[1]};
        ane_rewire(kv, ins, NULL);
    }
    {
        IOSurfaceRef ins[2] = {NULL, kq->ioInputs[1]};
        ane_rewire(kadd, ins, NULL);
    }

    a->y_surf = kadd->ioOutputs[0];
    return a;
}

// Set all 4 weight matrices. Each is [C,C] fp16 row-major.
static void ane_attn_set_weights(ANEAttn *a,
                                  const _Float16 *Wq, const _Float16 *Wk,
                                  const _Float16 *Wv, const _Float16 *Wo) {
    int CC = a->C * a->C;
    IOSurfaceLock(a->kq->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kq->ioInputs[0]), Wq, CC * sizeof(_Float16));
    IOSurfaceUnlock(a->kq->ioInputs[0], 0, NULL);

    IOSurfaceLock(a->kk->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kk->ioInputs[0]), Wk, CC * sizeof(_Float16));
    IOSurfaceUnlock(a->kk->ioInputs[0], 0, NULL);

    IOSurfaceLock(a->kv->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kv->ioInputs[0]), Wv, CC * sizeof(_Float16));
    IOSurfaceUnlock(a->kv->ioInputs[0], 0, NULL);

    IOSurfaceLock(a->kout->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kout->ioInputs[0]), Wo, CC * sizeof(_Float16));
    IOSurfaceUnlock(a->kout->ioInputs[0], 0, NULL);
}

// Chain prev's output to next's input (eliminate CPU roundtrip between layers)
static void ane_attn_chain(ANEAttn *prev, ANEAttn *next) {
    IOSurfaceRef y = prev->y_surf;
    next->x_surf = y;
    IOSurfaceRef ins[2] = {NULL, y};
    ane_rewire(next->kq, ins, NULL);
    ane_rewire(next->kk, ins, NULL);
    ane_rewire(next->kv, ins, NULL);
    ane_rewire(next->kadd, ins, NULL);
}

static void ane_attn_write_x(ANEAttn *a, const _Float16 *x) {
    IOSurfaceLock(a->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->x_surf), x, a->C * a->S * sizeof(_Float16));
    IOSurfaceUnlock(a->x_surf, 0, NULL);
}

static void ane_attn_eval(ANEAttn *a) {
    ane_eval(a->kq);
    ane_eval(a->kk);
    ane_eval(a->kv);
    ane_eval(a->kqk);
    ane_eval(a->kexp);
    ane_eval(a->krowsum);
    ane_eval(a->kdiv);
    ane_eval(a->kvattn);
    ane_eval(a->kout);
    ane_eval(a->kadd);
}

static void ane_attn_read_y(ANEAttn *a, _Float16 *y) {
    IOSurfaceLock(a->y_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(a->y_surf), a->C * a->S * sizeof(_Float16));
    IOSurfaceUnlock(a->y_surf, kIOSurfaceLockReadOnly, NULL);
}
