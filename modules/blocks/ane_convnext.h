// ane_convnext.h — ConvNeXt block (fully ANE: LN, dw, pw1, SiLU, pw2, add)
//
// Forward: LN → dw → LN → pw1(C→4C) → SiLU → pw2(4C→C) → add(residual)
//
// CPU work (unavoidable):
//   - 2× blocked transpose [C,S]↔[S,C] for dw layout change (~0.6ms at S=16384)
//   - K*K shift writes into dw IOSurfaces (~0.36ms at S=1024, scales with S)
//   - write_input memcpy (residual copy)
//
// Layout: [1, C, S] throughout (S = H*H spatial tokens)
//
// Usage:
//   ANEConvNeXt *blk = ane_convnext_compile(C, S, K);
//   ane_convnext_set_weights(blk, dw_fp16, pw1_fp16, pw2_fp16);
//   ane_convnext_write_input(blk, x_fp16);  // [C*S]
//   ane_convnext_eval(blk, H);              // H = sqrt(S)
//   ane_convnext_read_output(blk, y_fp16);  // [C*S]
//
#pragma once
#include "../ops/ane_ln.h"
#include "../ops/ane_matmul.h"
#include "../ops/ane_silu.h"
#include "../ops/ane_add.h"
#include "../ops/ane_dw.h"
#include "../ops/ane_fused_silu_pw2_add.h"
#include "../ops/ane_fused_pw1_silu_pw2_add.h"
#include <arm_neon.h>

typedef struct {
    int C, S, K;
    ANELayerNorm  *ln1;
    ANELayerNorm  *ln2;
    ANEMatmul     *pw1;   // C → 4C
    ANESilu       *silu;
    ANEMatmul     *pw2;   // 4C → C
    ANEAdd        *add;
    ANEFusedSiluPw2Add *fused_spa;  // fused silu+pw2+add (NULL = fallback)
    ANEFusedPw1SiluPw2Add *fused_all; // fused pw1+silu+pw2+add (NULL = fallback)
    // CPU dw
    _Float16 *dw_w;    // [C, K*K]
    _Float16 *dw_in;   // [C*S]
    _Float16 *dw_out;  // [C*S]
} ANEConvNeXt;

// Blocked transpose [C,S] → [S,C], 32×32 tiles for cache friendliness
static __attribute__((noinline))
void _cnx_transpose_cs_sc(const _Float16 *src, _Float16 *dst, int C, int S) {
    for (int c=0;c<C;c+=32) for (int s=0;s<S;s+=32) {
        int ce=c+32<C?c+32:C, se=s+32<S?s+32:S;
        for (int ci=c;ci<ce;ci++) for (int si=s;si<se;si++)
            dst[si*C+ci] = src[ci*S+si];
    }
}

// Blocked transpose [S,C] → [C,S]
// Tile over S outer, C inner: reads src[si*C+ci] sequential in ci (good),
// writes dst[ci*S+si] sequential in ci (stride S — scattered, but write-combining helps)
static __attribute__((noinline))
void _cnx_transpose_sc_cs(const _Float16 *src, _Float16 *dst, int C, int S) {
    for (int s=0;s<S;s+=32) for (int c=0;c<C;c+=32) {
        int se=s+32<S?s+32:S, ce=c+32<C?c+32:C;
        for (int si=s;si<se;si++) {
            const _Float16 *row = src + si*C + c;
            for (int ci=c;ci<ce;ci++)
                dst[ci*S+si] = row[ci-c];
        }
    }
}

// NEON-vectorized depthwise KxK same-pad, fp16 in/out, layout [C, H, H]
// Accumulates in fp32 scratch to avoid K*K fp16 read-modify-write cycles.
// Parallel over C — each channel is fully independent.
#include <dispatch/dispatch.h>
static void __attribute__((noinline))
_dw_neon_fp16(const _Float16 *x, const _Float16 *w, _Float16 *y,
              int C, int H, int K) {
    int S = H * H, pad = K / 2;
    dispatch_apply((size_t)C, DISPATCH_APPLY_AUTO, ^(size_t c_) {
        int c = (int)c_;
        const _Float16 *xc = x + c*S;
        _Float16 *yc = y + c*S;
        // fp32 accumulator for this channel — avoids K*K fp16 RMW cycles
        float acc_buf[S];
        memset(acc_buf, 0, (size_t)S * sizeof(float));
        for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            float wv = (float)w[c*K*K + ky*K + kx];
            float32x4_t vw = vdupq_n_f32(wv);
            for (int oh = 0; oh < H; oh++) {
                int ih = oh + ky - pad;
                if (ih < 0 || ih >= H) continue;
                int dx = kx - pad;
                int ow_start = (-dx > 0) ? -dx : 0;
                int ow_end = (H - dx < H) ? H - dx : H;
                int ow = ow_start;
                float *ac = acc_buf + oh*H;
                for (; ow + 7 < ow_end; ow += 8) {
                    int iw = ow + dx;
                    float16x8_t xh = vld1q_f16((const __fp16*)(xc + ih*H + iw));
                    float32x4_t xl = vcvt_f32_f16(vget_low_f16(xh));
                    float32x4_t xhi = vcvt_f32_f16(vget_high_f16(xh));
                    vst1q_f32(ac+ow,   vfmaq_f32(vld1q_f32(ac+ow),   xl,  vw));
                    vst1q_f32(ac+ow+4, vfmaq_f32(vld1q_f32(ac+ow+4), xhi, vw));
                }
                for (; ow < ow_end; ow++) {
                    int iw = ow + dx;
                    ac[ow] += wv * (float)xc[ih*H + iw];
                }
            }
        }
        // Convert fp32 accumulator → fp16 output (one pass)
        int s = 0;
        for (; s + 7 < S; s += 8) {
            vst1q_f16((__fp16*)(yc+s),
                vcombine_f16(vcvt_f16_f32(vld1q_f32(acc_buf+s)),
                             vcvt_f16_f32(vld1q_f32(acc_buf+s+4))));
        }
        for (; s < S; s++) yc[s] = (_Float16)acc_buf[s];
    });
}

static void _rewire(ANEKernel *k, int slot, IOSurfaceRef surf) {
    IOSurfaceRef ins[4] = {0}; ins[slot] = surf;
    ane_rewire(k, ins, NULL);
}

static ANEConvNeXt *ane_convnext_compile(int C, int S, int K) {
    ANEConvNeXt *blk = (ANEConvNeXt *)calloc(1, sizeof(ANEConvNeXt));
    blk->C = C; blk->S = S; blk->K = K;

    blk->dw_w   = (_Float16 *)calloc(C * K*K, sizeof(_Float16));
    blk->dw_in  = (_Float16 *)calloc(C * S,   sizeof(_Float16));
    blk->dw_out = (_Float16 *)calloc(C * S,   sizeof(_Float16));

    blk->ln1  = ane_ln_compile(C, S);
    blk->ln2  = ane_ln_compile(C, S);
    blk->pw1  = ane_matmul_compile(C, C*4, S);
    blk->silu = ane_silu_compile(C*4, S);
    blk->pw2  = ane_matmul_compile(C*4, C, S);
    blk->add  = ane_add_compile(C, S);

    if (!blk->ln1||!blk->ln2||!blk->pw1||!blk->silu||!blk->pw2||!blk->add) {
        fprintf(stderr, "ane_convnext_compile FAILED (C=%d S=%d)\n", C, S);
        return NULL;
    }

    // Wire ANE chain: pw1.x ← ln2.out
    _rewire(blk->pw1->k, 1, blk->ln2->out);
    blk->pw1->x_surf = blk->ln2->out;
    // silu.x ← pw1.y
    IOSurfaceRef silu_ins[2] = {blk->pw1->y_surf, 0};
    ane_rewire(blk->silu->k, silu_ins, NULL);
    blk->silu->x_surf = blk->pw1->y_surf;
    // pw2.x ← silu.y
    _rewire(blk->pw2->k, 1, blk->silu->y_surf);
    blk->pw2->x_surf = blk->silu->y_surf;
    // add.a ← pw2.y
    ane_add_rewire_a(blk->add, blk->pw2->y_surf);

    // Try fused silu+pw2+add (replaces 3 dispatches with 1)
    blk->fused_spa = ane_fused_silu_pw2_add_compile(C, S);
    if (blk->fused_spa) {
        // Wire: W ← pw2.w_surf, x ← pw1.y_surf (pre-silu), residual ← add.b_surf
        IOSurfaceRef fins[3] = {blk->pw2->w_surf, blk->add->b_surf, blk->pw1->y_surf};
        ane_rewire(blk->fused_spa->k, fins, NULL);
        blk->fused_spa->w_surf = blk->pw2->w_surf;
        blk->fused_spa->residual_surf = blk->add->b_surf;
        blk->fused_spa->x_surf = blk->pw1->y_surf;
    }

    // Try fused pw1+silu+pw2+add (replaces pw1 + fused_spa = 2 dispatches → 1)
    blk->fused_all = ane_fused_pw1_silu_pw2_add_compile(C, S);
    if (blk->fused_all) {
        // Wire: W_pw1 ← pw1.w_surf, W_pw2 ← pw2.w_surf,
        //        residual ← add.b_surf, x ← ln2.out
        IOSurfaceRef fins[4] = {blk->pw1->w_surf, blk->pw2->w_surf,
                                blk->add->b_surf, blk->ln2->out};
        ane_rewire(blk->fused_all->k, fins, NULL);
        blk->fused_all->w_pw1_surf    = blk->pw1->w_surf;
        blk->fused_all->w_pw2_surf    = blk->pw2->w_surf;
        blk->fused_all->residual_surf = blk->add->b_surf;
        blk->fused_all->x_surf        = blk->ln2->out;
    }

    return blk;
}

static IOSurfaceRef ane_convnext_output_surf(ANEConvNeXt *blk) {
    if (blk->fused_all) return blk->fused_all->y_surf;
    if (blk->fused_spa) return blk->fused_spa->y_surf;
    return blk->add->c_surf;
}

// Chain block `prev` output directly into block `next` input (no CPU memcpy).
// prev's output becomes next's ln1 input AND next's residual.
// Call after compiling both blocks, before any eval.
static void ane_convnext_chain(ANEConvNeXt *prev, ANEConvNeXt *next) {
    IOSurfaceRef out = ane_convnext_output_surf(prev);
    ane_ln_rewire_input(next->ln1, out);
    // Wire next's residual: fused_all, fused_spa, or add->b
    if (next->fused_all) {
        IOSurfaceRef fins[4] = {NULL, NULL, out, NULL};
        ane_rewire(next->fused_all->k, fins, NULL);
        next->fused_all->residual_surf = out;
    }
    if (next->fused_spa) {
        IOSurfaceRef fins[3] = {NULL, out, NULL};
        ane_rewire(next->fused_spa->k, fins, NULL);
        next->fused_spa->residual_surf = out;
    }
    ane_add_rewire_b(next->add, out);
}

static void ane_convnext_set_weights(ANEConvNeXt *blk,
                                      const _Float16 *dw,   // [C, K*K]
                                      const _Float16 *pw1,  // [4C, C]
                                      const _Float16 *pw2)  // [C, 4C]
{
    memcpy(blk->dw_w, dw, blk->C * blk->K*blk->K * sizeof(_Float16));
    ane_matmul_write_w(blk->pw1, pw1);
    ane_matmul_write_w(blk->pw2, pw2);
}

static void ane_convnext_write_input(ANEConvNeXt *blk, const _Float16 *x) {
    ane_ln_write_input(blk->ln1, x);
    // residual copy — goes to fused_all, fused_spa, or add->b_surf
    IOSurfaceRef res = blk->fused_all ? blk->fused_all->residual_surf :
                       blk->fused_spa ? blk->fused_spa->residual_surf : blk->add->b_surf;
    IOSurfaceLock(res, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(res), x, blk->C * blk->S * sizeof(_Float16));
    IOSurfaceUnlock(res, 0, NULL);
}

static double _cnxf_now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e3+ts.tv_nsec*1e-6;}
static int _cnxf_profile_step = 0;

static void ane_convnext_eval(ANEConvNeXt *blk, int H) {
    int _p = (++_cnxf_profile_step == 2 && blk->C == 96 && blk->S == 16384);
    double _t0, _t1;
#define _CFT(label) if(_p){_t1=_cnxf_now_ms();printf("    fwd_cnx %-16s %.1f ms\n",label,_t1-_t0);_t0=_t1;}
    if(_p) _t0=_cnxf_now_ms();

    // 1. LN1 (ANE)
    ane_ln_eval(blk->ln1);
    if(_p){_CFT("ln1");}

    // 2. Read ln1 output → NEON depthwise → write to ln2 input
    IOSurfaceLock(blk->ln1->out, kIOSurfaceLockReadOnly, NULL);
    memcpy(blk->dw_in, IOSurfaceGetBaseAddress(blk->ln1->out), blk->C * blk->S * sizeof(_Float16));
    IOSurfaceUnlock(blk->ln1->out, kIOSurfaceLockReadOnly, NULL);
    if(_p){_CFT("ln1_read+dw_in");}

    _dw_neon_fp16(blk->dw_in, blk->dw_w, blk->dw_out, blk->C, H, blk->K);
    if(_p){_CFT("dw_fwd");}

    ane_ln_write_input(blk->ln2, blk->dw_out);
    if(_p){_CFT("ln2_write");}

    // 3. LN2 → pointwise chain (ANE)
    ane_ln_eval(blk->ln2);    if(_p){_CFT("ln2");}
    if (blk->fused_all) {
        ane_fused_pw1_silu_pw2_add_eval(blk->fused_all);
        if(_p){_CFT("pw1+silu+pw2+add");}
    } else {
        ane_matmul_eval(blk->pw1); if(_p){_CFT("pw1");}
        if (blk->fused_spa) {
            ane_fused_silu_pw2_add_eval(blk->fused_spa);
            if(_p){_CFT("silu+pw2+add");}
        } else {
            ane_silu_eval(blk->silu);  if(_p){_CFT("silu");}
            ane_matmul_eval(blk->pw2); if(_p){_CFT("pw2");}
            ane_add_eval(blk->add);    if(_p){_CFT("add");}
        }
    }
#undef _CFT
}

static void ane_convnext_read_output(ANEConvNeXt *blk, _Float16 *y) {
    IOSurfaceRef out = ane_convnext_output_surf(blk);
    IOSurfaceLock(out, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(out), blk->C * blk->S * sizeof(_Float16));
    IOSurfaceUnlock(out, kIOSurfaceLockReadOnly, NULL);
}
