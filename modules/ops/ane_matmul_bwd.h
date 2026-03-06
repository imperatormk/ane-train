// ane_matmul_bwd.h — ANE matmul backward: dX and dW
//
// Forward:  Y [Co,S] = W [Co,Ci] @ X [Ci,S]
// Backward:
//   dX [Ci,S]  = W^T [Ci,Co] @ dY [Co,S]
//   dW [Co,Ci] = dY [Co,S]  @ X^T [S,Ci]
//
// CPU fallback activates automatically when ANE tensors exceed size limits.
//
#pragma once
#include "../../ane_runtime.h"
#include "../../mil_gen.h"

typedef struct {
    int Ci, Co, S;
    int cpu_mode;
    // ANE path
    ANEKernel *k_dx;
    IOSurfaceRef w_surf;
    IOSurfaceRef dy_surf;
    IOSurfaceRef dx_surf;
    ANEKernel *k_dw;
    IOSurfaceRef x_surf;
    IOSurfaceRef dw_surf;
    // CPU path — IOSurfaces used as shared buffers so rewire_w/rewire_x still work
    IOSurfaceRef cpu_w_surf;
    IOSurfaceRef cpu_dy_surf;
    IOSurfaceRef cpu_dx_surf;
    IOSurfaceRef cpu_x_surf;
    IOSurfaceRef cpu_dw_surf;
} ANEMatmulBwd;

// CPU: dX[Ci,S] = W^T[Ci,Co] @ dY[Co,S]
// NEON + parallel: split Co outer loop into 4 chunks, accumulate into fp32, convert at end.
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#define _MM_NTHREADS 4
static void _mm_bwd_dx_cpu(ANEMatmulBwd *bwd) {
    int Ci=bwd->Ci, Co=bwd->Co, S=bwd->S;
    IOSurfaceLock(bwd->w_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd->dy_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd->dx_surf, 0, NULL);
    const _Float16 *W  = (_Float16*)IOSurfaceGetBaseAddress(bwd->w_surf);
    const _Float16 *dy = (_Float16*)IOSurfaceGetBaseAddress(bwd->dy_surf);
    _Float16       *dx = (_Float16*)IOSurfaceGetBaseAddress(bwd->dx_surf);

    // fp32 partial sums: [_MM_NTHREADS, Ci, S] — each thread writes its own slab
    float *partials = (float *)calloc((size_t)_MM_NTHREADS * Ci * S, sizeof(float));

    // Each thread handles Co/N co-rows, accumulates into its own partial[Ci,S]
    dispatch_apply(_MM_NTHREADS, DISPATCH_APPLY_AUTO, ^(size_t tid) {
        int co0 = (int)tid * Co / _MM_NTHREADS;
        int co1 = ((int)tid+1) * Co / _MM_NTHREADS;
        float *part = partials + (int)tid * Ci * S;
        for (int co=co0; co<co1; co++) {
            const _Float16 *dy_co = dy + co*S;
            const _Float16 *W_co  = W  + co*Ci;
            for (int ci=0; ci<Ci; ci++) {
                float w = (float)W_co[ci];
                float32x4_t vw = vdupq_n_f32(w);
                float *p = part + ci*S;
                int s = 0;
                for (; s <= S-8; s += 8) {
                    vst1q_f32(p+s,   vmlaq_f32(vld1q_f32(p+s),   vw, vcvt_f32_f16(vld1_f16((const __fp16*)(dy_co+s)))));
                    vst1q_f32(p+s+4, vmlaq_f32(vld1q_f32(p+s+4), vw, vcvt_f32_f16(vld1_f16((const __fp16*)(dy_co+s+4)))));
                }
                for (; s<S; s++) p[s] += w*(float)dy_co[s];
            }
        }
    });

    // Reduce partials → dx (parallel over ci)
    dispatch_apply((size_t)Ci, DISPATCH_APPLY_AUTO, ^(size_t ci) {
        _Float16 *out = dx + (int)ci*S;
        int s = 0;
        for (; s <= S-4; s += 4) {
            float32x4_t acc = vdupq_n_f32(0.f);
            for (int tid=0; tid<_MM_NTHREADS; tid++)
                acc = vaddq_f32(acc, vld1q_f32(partials + tid*Ci*S + (int)ci*S + s));
            vst1_f16((__fp16*)(out+s), vcvt_f16_f32(acc));
        }
        for (; s<S; s++) {
            float acc=0.f;
            for (int tid=0; tid<_MM_NTHREADS; tid++)
                acc += partials[tid*Ci*S + (int)ci*S + s];
            out[s] = (_Float16)acc;
        }
    });

    free(partials);
    IOSurfaceUnlock(bwd->dx_surf, 0, NULL);
    IOSurfaceUnlock(bwd->dy_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(bwd->w_surf,  kIOSurfaceLockReadOnly, NULL);
}

// CPU: dW[Co,Ci] = dY[Co,S] @ X^T[S,Ci]
// NEON + parallel: split co across threads, NEON for inner s loop.
static void _mm_bwd_dw_cpu(ANEMatmulBwd *bwd) {
    int Ci=bwd->Ci, Co=bwd->Co, S=bwd->S;
    IOSurfaceLock(bwd->dy_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd->x_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd->dw_surf, 0, NULL);
    const _Float16 *dy = (_Float16*)IOSurfaceGetBaseAddress(bwd->dy_surf);
    const _Float16 *x  = (_Float16*)IOSurfaceGetBaseAddress(bwd->x_surf);
    _Float16       *dw = (_Float16*)IOSurfaceGetBaseAddress(bwd->dw_surf);

    dispatch_apply((size_t)Co, DISPATCH_APPLY_AUTO, ^(size_t co) {
        const _Float16 *dy_co = dy + (int)co*S;
        _Float16 *dw_co = dw + (int)co*Ci;
        for (int ci = 0; ci < Ci; ci++) {
            const _Float16 *x_ci = x + ci*S;
            float32x4_t vacc = vdupq_n_f32(0.f);
            int s = 0;
            for (; s <= S-4; s += 4)
                vacc = vmlaq_f32(vacc,
                    vcvt_f32_f16(vld1_f16((const __fp16*)(dy_co+s))),
                    vcvt_f32_f16(vld1_f16((const __fp16*)(x_ci+s))));
            float acc = vaddvq_f32(vacc);
            for (; s < S; s++) acc += (float)dy_co[s] * (float)x_ci[s];
            dw_co[ci] = (_Float16)acc;
        }
    });

    IOSurfaceUnlock(bwd->dw_surf, 0, NULL);
    IOSurfaceUnlock(bwd->x_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(bwd->dy_surf, kIOSurfaceLockReadOnly, NULL);
}

static ANEMatmulBwd *ane_matmul_bwd_compile(int Ci, int Co, int S) {
    size_t sw  = (size_t)Co * Ci * 2;  if (sw  < 2048) sw  = 2048;
    size_t sx  = (size_t)Ci * S  * 2;  if (sx  < 2048) sx  = 2048;
    size_t sdy = (size_t)Co * S  * 2;  if (sdy < 2048) sdy = 2048;
    size_t sdx = (size_t)Ci * S  * 2;  if (sdx < 2048) sdx = 2048;
    size_t sdw = (size_t)Co * Ci * 2;  if (sdw < 2048) sdw = 2048;

    size_t ins_dx[2] = {sw, sdy};
    ANEKernel *k_dx = ane_compile(
        [mil_gen_matmul_dx(Ci, Co, S) dataUsingEncoding:NSUTF8StringEncoding],
        nil, 2, ins_dx, 1, &sdx);

    size_t ins_dw[2] = {sdy, sx};
    ANEKernel *k_dw = k_dx ? ane_compile(
        [mil_gen_dW(Ci, Co, S) dataUsingEncoding:NSUTF8StringEncoding],
        nil, 2, ins_dw, 1, &sdw) : NULL;

    ANEMatmulBwd *bwd = (ANEMatmulBwd *)calloc(1, sizeof(ANEMatmulBwd));
    bwd->Ci = Ci; bwd->Co = Co; bwd->S = S;

    if (k_dx && k_dw) {
        bwd->k_dx = k_dx;
        bwd->k_dw = k_dw;
        bwd->w_surf  = k_dx->ioInputs[0];
        bwd->dy_surf = k_dx->ioInputs[1];
        bwd->dx_surf = k_dx->ioOutputs[0];
        IOSurfaceRef dw_ins[2] = {bwd->dy_surf, NULL};
        ane_rewire(k_dw, dw_ins, NULL);
        bwd->x_surf  = k_dw->ioInputs[1];
        bwd->dw_surf = k_dw->ioOutputs[0];
    } else {
        // CPU fallback: allocate IOSurfaces as shared buffers
        fprintf(stderr, "ane_matmul_bwd_compile: CPU fallback (Ci=%d Co=%d S=%d)\n", Ci, Co, S);
        bwd->cpu_mode = 1;
        bwd->w_surf   = ane_create_surface(sw);
        bwd->dy_surf  = ane_create_surface(sdy);
        bwd->dx_surf  = ane_create_surface(sdx);
        bwd->x_surf   = ane_create_surface(sx);
        bwd->dw_surf  = ane_create_surface(sdw);
    }
    return bwd;
}

static void ane_matmul_bwd_rewire_w(ANEMatmulBwd *bwd, IOSurfaceRef w_surf) {
    if (bwd->cpu_mode) { bwd->w_surf = w_surf; return; }
    IOSurfaceRef ins[2] = {w_surf, NULL};
    ane_rewire(bwd->k_dx, ins, NULL);
    bwd->w_surf = w_surf;
}

static void ane_matmul_bwd_rewire_x(ANEMatmulBwd *bwd, IOSurfaceRef x_surf) {
    if (bwd->cpu_mode) { bwd->x_surf = x_surf; return; }
    IOSurfaceRef ins[2] = {NULL, x_surf};
    ane_rewire(bwd->k_dw, ins, NULL);
    bwd->x_surf = x_surf;
}

static void ane_matmul_bwd_write_dy(ANEMatmulBwd *bwd, const _Float16 *dy) {
    IOSurfaceLock(bwd->dy_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(bwd->dy_surf), dy, bwd->Co * bwd->S * sizeof(_Float16));
    IOSurfaceUnlock(bwd->dy_surf, 0, NULL);
}

static void ane_matmul_bwd_write_x(ANEMatmulBwd *bwd, const _Float16 *x) {
    IOSurfaceLock(bwd->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(bwd->x_surf), x, bwd->Ci * bwd->S * sizeof(_Float16));
    IOSurfaceUnlock(bwd->x_surf, 0, NULL);
}

// Fast path: Co=1 (head layer). NEON + parallel over ci.
static void _mm_bwd_co1_cpu(ANEMatmulBwd *bwd) {
    int Ci=bwd->Ci, S=bwd->S;
    IOSurfaceLock(bwd->w_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd->dy_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd->dx_surf, 0, NULL);
    IOSurfaceLock(bwd->x_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd->dw_surf, 0, NULL);
    const _Float16 *W  = (_Float16*)IOSurfaceGetBaseAddress(bwd->w_surf);
    const _Float16 *dy = (_Float16*)IOSurfaceGetBaseAddress(bwd->dy_surf);
    _Float16       *dx = (_Float16*)IOSurfaceGetBaseAddress(bwd->dx_surf);
    const _Float16 *x  = (_Float16*)IOSurfaceGetBaseAddress(bwd->x_surf);
    _Float16       *dw = (_Float16*)IOSurfaceGetBaseAddress(bwd->dw_surf);

    dispatch_apply((size_t)Ci, DISPATCH_APPLY_AUTO, ^(size_t ci_) {
        int ci = (int)ci_;
        float w = (float)W[ci];
        float32x4_t vw = vdupq_n_f32(w);
        _Float16 *dx_ci = dx + ci*S;
        const _Float16 *x_ci = x + ci*S;
        float32x4_t vacc = vdupq_n_f32(0.f);
        int s = 0;
        for (; s <= S-8; s += 8) {
            float16x4_t d0 = vld1_f16((const __fp16*)(dy+s));
            float16x4_t d1 = vld1_f16((const __fp16*)(dy+s+4));
            // dX
            vst1_f16((__fp16*)(dx_ci+s),   vcvt_f16_f32(vmulq_f32(vw, vcvt_f32_f16(d0))));
            vst1_f16((__fp16*)(dx_ci+s+4), vcvt_f16_f32(vmulq_f32(vw, vcvt_f32_f16(d1))));
            // dW accumulate
            vacc = vmlaq_f32(vacc, vcvt_f32_f16(d0),
                             vcvt_f32_f16(vld1_f16((const __fp16*)(x_ci+s))));
            vacc = vmlaq_f32(vacc, vcvt_f32_f16(d1),
                             vcvt_f32_f16(vld1_f16((const __fp16*)(x_ci+s+4))));
        }
        float acc = vaddvq_f32(vacc);
        for (; s < S; s++) {
            dx_ci[s] = (_Float16)(w * (float)dy[s]);
            acc += (float)dy[s] * (float)x_ci[s];
        }
        dw[ci] = (_Float16)acc;
    });

    IOSurfaceUnlock(bwd->dw_surf, 0, NULL);
    IOSurfaceUnlock(bwd->x_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(bwd->dx_surf, 0, NULL);
    IOSurfaceUnlock(bwd->dy_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(bwd->w_surf,  kIOSurfaceLockReadOnly, NULL);
}

static void ane_matmul_bwd_eval(ANEMatmulBwd *bwd) {
    if (bwd->cpu_mode) {
        if (bwd->Co == 1) { _mm_bwd_co1_cpu(bwd); return; }
        _mm_bwd_dx_cpu(bwd); _mm_bwd_dw_cpu(bwd); return;
    }
    ane_eval(bwd->k_dx);
    ane_eval(bwd->k_dw);
}

static void ane_matmul_bwd_read_dx(ANEMatmulBwd *bwd, _Float16 *dx) {
    IOSurfaceLock(bwd->dx_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(dx, IOSurfaceGetBaseAddress(bwd->dx_surf), bwd->Ci * bwd->S * sizeof(_Float16));
    IOSurfaceUnlock(bwd->dx_surf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_matmul_bwd_read_dw(ANEMatmulBwd *bwd, _Float16 *dw) {
    IOSurfaceLock(bwd->dw_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(dw, IOSurfaceGetBaseAddress(bwd->dw_surf), bwd->Co * bwd->Ci * sizeof(_Float16));
    IOSurfaceUnlock(bwd->dw_surf, kIOSurfaceLockReadOnly, NULL);
}
