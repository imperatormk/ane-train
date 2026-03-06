// ane_mhattn.h — Multi-head self-attention on ANE
//
// Layout: [1, C, S] throughout
//
// Pipeline:
//   QKV proj (ANE):  Wq/Wk/Wv[C,C] @ x[C,S] → Q,K,V[C,S]   (3 kernels, shared x_surf)
//   CPU split:       Q,K,V[C,S] → Q_h,K_h,V_h[d,S] for h=0..nH-1  (strided memcpy)
//   Per-head (ANE):  scores = Q_h^T @ K_h / sqrt(d) → softmax → V_h @ attn → out_h[d,S]
//                    (nH × 5 kernels: kqk, kexp, krowsum, kdiv, kvattn)
//   CPU concat:      out_h[d,S] → out[C,S]
//   Out proj (ANE):  Wo[C,C] @ out[C,S] → [C,S]
//   Residual (ANE):  + x → y[C,S]
//
// Constraints:
//   - d = C/nH must be multiple of 32
//   - C must be multiple of 32 (implied)
//   - C <= S (slot rule for QKV proj and out proj)
//   - d <= S (slot rule for per-head score matmul — almost always true)
//
// Usage:
//   ANEMHAttn *mha = ane_mhattn_compile(C, nH, S);
//   ane_mhattn_set_weights(mha, Wq, Wk, Wv, Wo);  // each [C*C] fp16
//   ane_mhattn_write_x(mha, x_fp16);               // [C*S]
//   ane_mhattn_eval(mha);
//   ane_mhattn_read_y(mha, y_fp16);                // [C*S]
//
#pragma once
#include "../../ane_runtime.h"
#include <string.h>

#define ANE_MHATTN_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]\n"

// Per-head kernel set (5 kernels for score+softmax+vattn)
typedef struct {
    ANEKernel *kqk;      // scores = Q_h^T @ K_h / sqrt(d)
    ANEKernel *kexp;     // exp(scores)
    ANEKernel *krowsum;  // rowsum = ones[1,S] @ exp^T
    ANEKernel *kdiv;     // attn = exp / rowsum
    ANEKernel *kvattn;   // out_h = V_h @ attn
    // IOSurfaces for head inputs (written by CPU split)
    IOSurfaceRef q_surf; // Q_h[d,S] — slot0 of kqk
    IOSurfaceRef k_surf; // K_h[d,S] — slot1 of kqk
    IOSurfaceRef v_surf; // V_h[d,S] — slot0 of kvattn
} ANEMHHead;

typedef struct {
    int C, nH, d, S;    // d = C/nH (head dim)
    // QKV projection kernels (shared x_surf)
    ANEKernel *kq, *kk, *kv;
    // Out proj + residual
    ANEKernel *kout, *kadd;
    // Per-head kernels
    ANEMHHead *heads;   // [nH]
    // IOSurfaces
    IOSurfaceRef x_surf;    // input [C,S]
    IOSurfaceRef y_surf;    // output [C,S]
    IOSurfaceRef q_full;    // Q[C,S] = kq->ioOutputs[0]
    IOSurfaceRef k_full;    // K[C,S] = kk->ioOutputs[0]
    IOSurfaceRef v_full;    // V[C,S] = kv->ioOutputs[0]
    // CPU scratch buffers
    _Float16 *cat_buf;      // concat of head outputs [C,S]
} ANEMHAttn;

// --- MIL generators (parameterized by d=head_dim, S) ---

static NSString *_mha_mil_proj(int C, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_MHATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> W, tensor<fp16,[1,%d,%d]> X) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,%d,%d]> Y=matmul(transpose_x=ff,transpose_y=ff,x=W,y=X)"
        "[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        C,C, C,S, C,S];
}

static NSString *_mha_mil_scores(int d, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_MHATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> Q, tensor<fp16,[1,%d,%d]> K) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<fp16,[]> SC=const()[name=tensor<string,[]>(\"SC\"),val=tensor<fp16,[]>(%f)];\n"
        "    tensor<fp16,[1,%d,%d]> RAW=matmul(transpose_x=tt,transpose_y=ff,x=Q,y=K)"
        "[name=tensor<string,[]>(\"RAW\")];\n"
        "    tensor<fp16,[1,%d,%d]> SCORES=mul(x=RAW,y=SC)[name=tensor<string,[]>(\"SCORES\")];\n"
        "  } -> (SCORES);\n}\n",
        d,S, d,S, 1.0f/sqrtf((float)d), S,S, S,S];
}

static NSString *_mha_mil_exp(int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_MHATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> X) {\n"
        "    tensor<fp16,[1,%d,%d]> Y=exp(x=X)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        S,S, S,S];
}

static NSString *_mha_mil_rowsum(int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_MHATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,1,%d]> ONES, tensor<fp16,[1,%d,%d]> ESC) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<bool,[]> tt=const()[name=tensor<string,[]>(\"tt\"),val=tensor<bool,[]>(true)];\n"
        "    tensor<fp16,[1,1,%d]> RS=matmul(transpose_x=ff,transpose_y=tt,x=ONES,y=ESC)"
        "[name=tensor<string,[]>(\"RS\")];\n"
        "  } -> (RS);\n}\n",
        S, S,S, S];
}

static NSString *_mha_mil_div(int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_MHATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,1,%d]> RS, tensor<fp16,[1,%d,%d]> ESC) {\n"
        "    tensor<fp16,[1,%d,%d]> ATTN=real_div(x=ESC,y=RS)[name=tensor<string,[]>(\"ATTN\")];\n"
        "  } -> (ATTN);\n}\n",
        S, S,S, S,S];
}

static NSString *_mha_mil_vattn(int d, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_MHATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> V, tensor<fp16,[1,%d,%d]> ATTN) {\n"
        "    tensor<bool,[]> ff=const()[name=tensor<string,[]>(\"ff\"),val=tensor<bool,[]>(false)];\n"
        "    tensor<fp16,[1,%d,%d]> OUT=matmul(transpose_x=ff,transpose_y=ff,x=V,y=ATTN)"
        "[name=tensor<string,[]>(\"OUT\")];\n"
        "  } -> (OUT);\n}\n",
        d,S, S,S, d,S];
}

static NSString *_mha_mil_add(int C, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n" ANE_MHATTN_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> A, tensor<fp16,[1,%d,%d]> B) {\n"
        "    tensor<fp16,[1,%d,%d]> Y=add(x=A,y=B)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        C,S, C,S, C,S];
}

static ANEMHAttn *ane_mhattn_compile(int C, int nH, int S) {
    int d = C / nH;
    if (C % nH != 0) {
        fprintf(stderr, "ane_mhattn_compile: C=%d not divisible by nH=%d\n", C, nH);
        return NULL;
    }
    if (d % 32 != 0) {
        fprintf(stderr, "ane_mhattn_compile: head_dim d=%d must be multiple of 32 (C=%d nH=%d)\n", d, C, nH);
        return NULL;
    }
    if (C > S) {
        fprintf(stderr, "ane_mhattn_compile: C=%d must be <= S=%d\n", C, S);
        return NULL;
    }

    size_t sCC = (size_t)C*C*2; if (sCC<2048) sCC=2048;
    size_t sCS = (size_t)C*S*2; if (sCS<2048) sCS=2048;
    size_t sdS = (size_t)d*S*2; if (sdS<2048) sdS=2048;
    size_t sSS = (size_t)S*S*2; if (sSS<2048) sSS=2048;
    size_t sS1 = (size_t)S*1*2; if (sS1<2048) sS1=2048;

#define MHA_COMPILE(name, mil, ni, ins, no, outs) \
    ANEKernel *name = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, ni, ins, no, outs); \
    if (!name) { fprintf(stderr, "ane_mhattn_compile: " #name " FAILED (C=%d nH=%d S=%d)\n", C, nH, S); return NULL; }

    // QKV projections: W[C,C]=slot0, x[C,S]=slot1
    size_t ins_proj[2] = {sCC, sCS};
    MHA_COMPILE(kq, _mha_mil_proj(C,S), 2, ins_proj, 1, &sCS);
    MHA_COMPILE(kk, _mha_mil_proj(C,S), 2, ins_proj, 1, &sCS);
    MHA_COMPILE(kv, _mha_mil_proj(C,S), 2, ins_proj, 1, &sCS);

    // Out proj + residual
    size_t ins_add[2] = {sCS, sCS};
    MHA_COMPILE(kout, _mha_mil_proj(C,S), 2, ins_proj, 1, &sCS);
    MHA_COMPILE(kadd, _mha_mil_add(C,S),  2, ins_add,  1, &sCS);
#undef MHA_COMPILE

    // Per-head kernels
    ANEMHHead *heads = (ANEMHHead *)calloc(nH, sizeof(ANEMHHead));
    size_t ins_qk[2]  = {sdS, sdS};
    size_t ins_rs[2]  = {sS1, sSS};
    size_t ins_div[2] = {sS1, sSS};
    size_t ins_va[2]  = {sdS, sSS};

    for (int h = 0; h < nH; h++) {
#define HEAD_COMPILE(field, mil, ni, ins, no, outs) do { \
    heads[h].field = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, ni, ins, no, outs); \
    if (!heads[h].field) { \
        fprintf(stderr, "ane_mhattn_compile: head[%d]." #field " FAILED\n", h); \
        return NULL; \
    } } while(0)

        HEAD_COMPILE(kqk,    _mha_mil_scores(d,S), 2, ins_qk,  1, &sSS);
        HEAD_COMPILE(kexp,   _mha_mil_exp(S),      1, &sSS,    1, &sSS);
        HEAD_COMPILE(kdiv,   _mha_mil_div(S),      2, ins_div, 1, &sSS);
        HEAD_COMPILE(krowsum,_mha_mil_rowsum(S),   2, ins_rs,  1, &sS1);
        HEAD_COMPILE(kvattn, _mha_mil_vattn(d,S),  2, ins_va,  1, &sdS);
#undef HEAD_COMPILE

        // Wire: kexp input ← kqk output
        { IOSurfaceRef ins[1]={heads[h].kqk->ioOutputs[0]}; ane_rewire(heads[h].kexp, ins, NULL); }
        // Wire: krowsum slot1 ← kexp output (slot0=ones, set below)
        { IOSurfaceRef ins[2]={NULL, heads[h].kexp->ioOutputs[0]}; ane_rewire(heads[h].krowsum, ins, NULL); }
        // Wire: kdiv slot0=krowsum->out, slot1=kexp->out
        { IOSurfaceRef ins[2]={heads[h].krowsum->ioOutputs[0], heads[h].kexp->ioOutputs[0]}; ane_rewire(heads[h].kdiv, ins, NULL); }
        // Wire: kvattn slot1=attn=kdiv->out
        { IOSurfaceRef ins[2]={NULL, heads[h].kdiv->ioOutputs[0]}; ane_rewire(heads[h].kvattn, ins, NULL); }

        // Fill ones surface
        IOSurfaceLock(heads[h].krowsum->ioInputs[0], 0, NULL);
        _Float16 *ones = (_Float16 *)IOSurfaceGetBaseAddress(heads[h].krowsum->ioInputs[0]);
        for (int i=0; i<S; i++) ones[i] = (_Float16)1.0f;
        IOSurfaceUnlock(heads[h].krowsum->ioInputs[0], 0, NULL);

        // Expose head input surfaces (written by CPU split)
        heads[h].q_surf = heads[h].kqk->ioInputs[0];   // Q_h slot0
        heads[h].k_surf = heads[h].kqk->ioInputs[1];   // K_h slot1
        heads[h].v_surf = heads[h].kvattn->ioInputs[0]; // V_h slot0
    }

    ANEMHAttn *a = (ANEMHAttn *)calloc(1, sizeof(ANEMHAttn));
    a->C = C; a->nH = nH; a->d = d; a->S = S;
    a->kq = kq; a->kk = kk; a->kv = kv;
    a->kout = kout; a->kadd = kadd;
    a->heads = heads;

    // x_surf: kq slot1; rewire kk/kv to same
    a->x_surf = kq->ioInputs[1];
    { IOSurfaceRef ins[2]={NULL, kq->ioInputs[1]}; ane_rewire(kk, ins, NULL); }
    { IOSurfaceRef ins[2]={NULL, kq->ioInputs[1]}; ane_rewire(kv, ins, NULL); }

    a->q_full = kq->ioOutputs[0];
    a->k_full = kk->ioOutputs[0];
    a->v_full = kv->ioOutputs[0];

    // kout slot1 ← cat_buf (written after CPU concat); exposed via kout->ioInputs[1]
    // kadd slot0 ← kout->out, slot1 ← x
    { IOSurfaceRef ins[2]={kout->ioOutputs[0], NULL}; ane_rewire(kadd, ins, NULL); }
    { IOSurfaceRef ins[2]={NULL, kq->ioInputs[1]}; ane_rewire(kadd, ins, NULL); }

    a->y_surf = kadd->ioOutputs[0];
    a->cat_buf = (_Float16 *)malloc((size_t)C * S * sizeof(_Float16));
    return a;
}

static void ane_mhattn_set_weights(ANEMHAttn *a,
                                    const _Float16 *Wq, const _Float16 *Wk,
                                    const _Float16 *Wv, const _Float16 *Wo) {
    int CC = a->C * a->C;
    IOSurfaceLock(a->kq->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kq->ioInputs[0]), Wq, CC*2);
    IOSurfaceUnlock(a->kq->ioInputs[0], 0, NULL);
    IOSurfaceLock(a->kk->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kk->ioInputs[0]), Wk, CC*2);
    IOSurfaceUnlock(a->kk->ioInputs[0], 0, NULL);
    IOSurfaceLock(a->kv->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kv->ioInputs[0]), Wv, CC*2);
    IOSurfaceUnlock(a->kv->ioInputs[0], 0, NULL);
    IOSurfaceLock(a->kout->ioInputs[0], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kout->ioInputs[0]), Wo, CC*2);
    IOSurfaceUnlock(a->kout->ioInputs[0], 0, NULL);
}

static void ane_mhattn_write_x(ANEMHAttn *a, const _Float16 *x) {
    IOSurfaceLock(a->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->x_surf), x, (size_t)a->C * a->S * 2);
    IOSurfaceUnlock(a->x_surf, 0, NULL);
}

// CPU: split Q[C,S], K[C,S], V[C,S] into per-head [d,S] surfaces
static void _mha_split(ANEMHAttn *a) {
    int d = a->d, S = a->S, nH = a->nH;
    // Read Q/K/V from ANE output surfaces
    _Float16 *Q = (_Float16 *)IOSurfaceGetBaseAddress(a->q_full);
    _Float16 *K = (_Float16 *)IOSurfaceGetBaseAddress(a->k_full);
    _Float16 *V = (_Float16 *)IOSurfaceGetBaseAddress(a->v_full);
    IOSurfaceLock(a->q_full, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(a->k_full, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(a->v_full, kIOSurfaceLockReadOnly, NULL);
    for (int h = 0; h < nH; h++) {
        // head h occupies rows [h*d .. (h+1)*d) of [C,S]
        const _Float16 *Qh = Q + (size_t)h*d*S;
        const _Float16 *Kh = K + (size_t)h*d*S;
        const _Float16 *Vh = V + (size_t)h*d*S;
        IOSurfaceLock(a->heads[h].q_surf, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(a->heads[h].q_surf), Qh, (size_t)d*S*2);
        IOSurfaceUnlock(a->heads[h].q_surf, 0, NULL);
        IOSurfaceLock(a->heads[h].k_surf, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(a->heads[h].k_surf), Kh, (size_t)d*S*2);
        IOSurfaceUnlock(a->heads[h].k_surf, 0, NULL);
        IOSurfaceLock(a->heads[h].v_surf, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(a->heads[h].v_surf), Vh, (size_t)d*S*2);
        IOSurfaceUnlock(a->heads[h].v_surf, 0, NULL);
    }
    IOSurfaceUnlock(a->q_full, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(a->k_full, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(a->v_full, kIOSurfaceLockReadOnly, NULL);
}

// CPU: concat head outputs [d,S] → cat_buf[C,S], then write to kout slot1
static void _mha_concat(ANEMHAttn *a) {
    int d = a->d, S = a->S, nH = a->nH;
    for (int h = 0; h < nH; h++) {
        IOSurfaceLock(a->heads[h].kvattn->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
        const _Float16 *src = (_Float16 *)IOSurfaceGetBaseAddress(a->heads[h].kvattn->ioOutputs[0]);
        memcpy(a->cat_buf + (size_t)h*d*S, src, (size_t)d*S*2);
        IOSurfaceUnlock(a->heads[h].kvattn->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    }
    // Write concatenated output to kout's x input (slot1)
    IOSurfaceLock(a->kout->ioInputs[1], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(a->kout->ioInputs[1]), a->cat_buf, (size_t)a->C*S*2);
    IOSurfaceUnlock(a->kout->ioInputs[1], 0, NULL);
}

static void ane_mhattn_eval(ANEMHAttn *a) {
    // 1. QKV projections
    ane_eval(a->kq);
    ane_eval(a->kk);
    ane_eval(a->kv);
    // 2. CPU split Q/K/V into head surfaces
    _mha_split(a);
    // 3. Per-head attention
    for (int h = 0; h < a->nH; h++) {
        ane_eval(a->heads[h].kqk);
        ane_eval(a->heads[h].kexp);
        ane_eval(a->heads[h].krowsum);
        ane_eval(a->heads[h].kdiv);
        ane_eval(a->heads[h].kvattn);
    }
    // 4. CPU concat head outputs → kout input
    _mha_concat(a);
    // 5. Out proj + residual
    ane_eval(a->kout);
    ane_eval(a->kadd);
}

static void ane_mhattn_read_y(ANEMHAttn *a, _Float16 *y) {
    IOSurfaceLock(a->y_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(a->y_surf), (size_t)a->C * a->S * 2);
    IOSurfaceUnlock(a->y_surf, kIOSurfaceLockReadOnly, NULL);
}
