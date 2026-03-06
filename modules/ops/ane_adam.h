// ane_adam.h — Adam optimizer (3 ANE kernels: k_m, k_v, k_w — proven pattern)
// ANE path: N <= ANE_ADAM_MAX. CPU fallback (NEON) for larger tensors.
//
// k_m: (dW[N], m[N]) → m_new[N]       2-in 1-out ✅
// k_v: (dW[N], v[N]) → v_new[N]       2-in 1-out ✅
// k_w: (W[N], m_new[N], v_new[N]) → W_new[N]  3-in 1-out ✅
//
#pragma once
#include "../../ane_runtime.h"
#include "../../mil_gen.h"
#include <arm_neon.h>

#define ANE_ADAM_MAX 16384

typedef struct {
    int N;
    int cpu_mode;
    float lr, beta1, beta2, eps;
    // ANE path
    ANEKernel *k_m;
    ANEKernel *k_v;
    ANEKernel *k_w;
    IOSurfaceRef dw_surf;
    IOSurfaceRef m_surf;
    IOSurfaceRef v_surf;
    IOSurfaceRef w_surf;
    IOSurfaceRef w_new_surf;
    // CPU path
    float *cpu_m, *cpu_v;
    IOSurfaceRef cpu_dw_surf;
    IOSurfaceRef cpu_w_surf;
    IOSurfaceRef cpu_w_new_surf;
} ANEAdam;

static ANEAdam *_ane_adam_compile_cpu(int N, float lr, float beta1, float beta2, float eps) {
    ANEAdam *opt = (ANEAdam *)calloc(1, sizeof(ANEAdam));
    opt->N = N; opt->cpu_mode = 1;
    opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    opt->cpu_m = (float *)calloc(N, sizeof(float));
    opt->cpu_v = (float *)calloc(N, sizeof(float));
    size_t sn = (size_t)N * 2; if (sn < 2048) sn = 2048;
    opt->cpu_dw_surf   = ane_create_surface(sn);
    opt->cpu_w_surf    = ane_create_surface(sn);
    opt->cpu_w_new_surf= ane_create_surface(sn);
    opt->dw_surf    = opt->cpu_dw_surf;
    opt->w_surf     = opt->cpu_w_surf;
    opt->w_new_surf = opt->cpu_w_new_surf;
    return opt;
}

static ANEAdam *ane_adam_compile(int N, float lr, float beta1, float beta2, float eps) {
    if (N > ANE_ADAM_MAX)
        return _ane_adam_compile_cpu(N, lr, beta1, beta2, eps);

    size_t sn = (size_t)N * 2; if (sn < 2048) sn = 2048;
    size_t ins2[2] = {sn, sn};
    size_t ins3[3] = {sn, sn, sn};

    ANEKernel *k_m = ane_compile([mil_gen_adam_m(N, beta1) dataUsingEncoding:NSUTF8StringEncoding],
                                  nil, 2, ins2, 1, &sn);
    ANEKernel *k_v = ane_compile([mil_gen_adam_v(N, beta2) dataUsingEncoding:NSUTF8StringEncoding],
                                  nil, 2, ins2, 1, &sn);
    float bc1 = 1.0f - beta1, bc2 = 1.0f - beta2;
    float lr_t1 = lr * sqrtf(bc2) / bc1;
    ANEKernel *k_w = ane_compile(
        [mil_gen_adam_w(N, lr_t1, fmaxf(eps, 1e-4f)) dataUsingEncoding:NSUTF8StringEncoding],
        nil, 3, ins3, 1, &sn);

    if (!k_m || !k_v || !k_w) {
        fprintf(stderr, "ane_adam_compile FAILED (N=%d)\n", N);
        return _ane_adam_compile_cpu(N, lr, beta1, beta2, eps);
    }

    ANEAdam *opt = (ANEAdam *)calloc(1, sizeof(ANEAdam));
    opt->N = N; opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    opt->k_m = k_m; opt->k_v = k_v; opt->k_w = k_w;

    opt->dw_surf = k_m->ioInputs[0];
    opt->m_surf  = k_m->ioInputs[1];

    // Wire dw into k_v slot 0
    IOSurfaceRef kv_ins[2] = {opt->dw_surf, NULL};
    ane_rewire(k_v, kv_ins, NULL);
    opt->v_surf = k_v->ioInputs[1];

    // Wire m_new, v_new into k_w
    IOSurfaceRef kw_ins[3] = {NULL, k_m->ioOutputs[0], k_v->ioOutputs[0]};
    ane_rewire(k_w, kw_ins, NULL);
    opt->w_surf     = k_w->ioInputs[0];
    opt->w_new_surf = k_w->ioOutputs[0];

    // Zero-init m and v
    IOSurfaceLock(opt->m_surf, 0, NULL);
    memset(IOSurfaceGetBaseAddress(opt->m_surf), 0, sn);
    IOSurfaceUnlock(opt->m_surf, 0, NULL);
    IOSurfaceLock(opt->v_surf, 0, NULL);
    memset(IOSurfaceGetBaseAddress(opt->v_surf), 0, sn);
    IOSurfaceUnlock(opt->v_surf, 0, NULL);

    return opt;
}

static void ane_adam_rewire_dw(ANEAdam *opt, IOSurfaceRef dw_surf) {
    if (opt->cpu_mode) { opt->dw_surf = dw_surf; return; }
    IOSurfaceRef ins_m[2] = {dw_surf, NULL};
    ane_rewire(opt->k_m, ins_m, NULL);
    IOSurfaceRef ins_v[2] = {dw_surf, NULL};
    ane_rewire(opt->k_v, ins_v, NULL);
    opt->dw_surf = dw_surf;
}

static void ane_adam_rewire_w(ANEAdam *opt, IOSurfaceRef w_surf) {
    if (opt->cpu_mode) { opt->w_surf = w_surf; return; }
    IOSurfaceRef ins[3] = {w_surf, NULL, NULL};
    ane_rewire(opt->k_w, ins, NULL);
    opt->w_surf = w_surf;
}

// CPU Adam — NEON vectorized
static void _ane_adam_step_cpu(ANEAdam *opt, int t) {
    float b1t = powf(opt->beta1, (float)t);
    float b2t = powf(opt->beta2, (float)t);
    float lr_t = opt->lr * sqrtf(1.0f - b2t) / (1.0f - b1t);
    float b1 = opt->beta1, b2 = opt->beta2, eps = opt->eps;
    float ob1 = 1.0f - b1, ob2 = 1.0f - b2;
    int N = opt->N;

    IOSurfaceLock(opt->dw_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(opt->w_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(opt->w_new_surf, 0, NULL);
    const _Float16 *dw_ptr = (const _Float16 *)IOSurfaceGetBaseAddress(opt->dw_surf);
    const _Float16 *w_ptr  = (const _Float16 *)IOSurfaceGetBaseAddress(opt->w_surf);
    _Float16       *wn_ptr = (_Float16 *)IOSurfaceGetBaseAddress(opt->w_new_surf);
    float *m = opt->cpu_m, *v = opt->cpu_v;

    float32x4_t vb1=vdupq_n_f32(b1), vob1=vdupq_n_f32(ob1);
    float32x4_t vb2=vdupq_n_f32(b2), vob2=vdupq_n_f32(ob2);
    float32x4_t vlr=vdupq_n_f32(lr_t), veps=vdupq_n_f32(eps);

    int i = 0;
    for (; i <= N-4; i += 4) {
        float32x4_t dw = vcvt_f32_f16(vld1_f16((const __fp16 *)(dw_ptr+i)));
        float32x4_t w  = vcvt_f32_f16(vld1_f16((const __fp16 *)(w_ptr+i)));
        float32x4_t mi = vld1q_f32(m+i), vi = vld1q_f32(v+i);
        mi = vmlaq_f32(vmulq_f32(vb1,mi), vob1, dw);
        vi = vmlaq_f32(vmulq_f32(vb2,vi), vob2, vmulq_f32(dw,dw));
        vst1q_f32(m+i, mi); vst1q_f32(v+i, vi);
        float32x4_t rs = vrsqrteq_f32(vi);
        rs = vmulq_f32(rs, vrsqrtsq_f32(vi, vmulq_f32(rs,rs)));
        float32x4_t sqv = vrecpeq_f32(rs);
        sqv = vmulq_f32(sqv, vrecpsq_f32(rs, sqv));
        float32x4_t denom = vaddq_f32(sqv, veps);
        float32x4_t rd = vrecpeq_f32(denom);
        rd = vmulq_f32(rd, vrecpsq_f32(denom, rd));
        float32x4_t wn = vsubq_f32(w, vmulq_f32(vlr, vmulq_f32(mi, rd)));
        vst1_f16((__fp16 *)(wn_ptr+i), vcvt_f16_f32(wn));
    }
    for (; i < N; i++) {
        float dw=(float)dw_ptr[i], w=(float)w_ptr[i];
        m[i] = b1*m[i] + ob1*dw;
        v[i] = b2*v[i] + ob2*dw*dw;
        wn_ptr[i] = (_Float16)(w - lr_t*m[i]/(sqrtf(v[i])+eps));
    }

    IOSurfaceUnlock(opt->w_new_surf, 0, NULL);
    IOSurfaceUnlock(opt->w_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(opt->dw_surf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_adam_step(ANEAdam *opt, int t) {
    if (opt->cpu_mode) { _ane_adam_step_cpu(opt, t); return; }
    (void)t;

    ane_eval(opt->k_m);
    ane_eval(opt->k_v);

    // Ping-pong m state
    IOSurfaceLock(opt->k_m->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(opt->m_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(opt->m_surf),
           IOSurfaceGetBaseAddress(opt->k_m->ioOutputs[0]),
           (size_t)opt->N * sizeof(_Float16));
    IOSurfaceUnlock(opt->m_surf, 0, NULL);
    IOSurfaceUnlock(opt->k_m->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);

    // Ping-pong v state
    IOSurfaceLock(opt->k_v->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(opt->v_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(opt->v_surf),
           IOSurfaceGetBaseAddress(opt->k_v->ioOutputs[0]),
           (size_t)opt->N * sizeof(_Float16));
    IOSurfaceUnlock(opt->v_surf, 0, NULL);
    IOSurfaceUnlock(opt->k_v->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);

    ane_eval(opt->k_w);
}
