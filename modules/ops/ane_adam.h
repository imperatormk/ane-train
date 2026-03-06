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
    float lr, beta1, beta2, eps, wd;
    float dw_clip;  // max |dW| per element (0 = disabled)
    // ANE path
    ANEKernel *k_m;
    ANEKernel *k_v;
    ANEKernel *k_w;
    IOSurfaceRef dw_surf;
    IOSurfaceRef m_surf;
    IOSurfaceRef v_surf;
    IOSurfaceRef w_surf;
    IOSurfaceRef w_new_surf;
    IOSurfaceRef lr_surf;   // runtime lr — written each step
    // CPU path
    float *cpu_m, *cpu_v;
    IOSurfaceRef cpu_dw_surf;
    IOSurfaceRef cpu_w_surf;
    IOSurfaceRef cpu_w_new_surf;
} ANEAdam;

static ANEAdam *_ane_adam_compile_cpu(int N, float lr, float beta1, float beta2, float eps, float wd) {
    ANEAdam *opt = (ANEAdam *)calloc(1, sizeof(ANEAdam));
    opt->N = N; opt->cpu_mode = 1;
    opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps; opt->wd = wd;
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

// shared_lr_surf: if non-NULL, all k_w kernels share this single IOSurface for lr.
// Write lr once per step to shared_lr_surf instead of per-optimizer.
// Pass NULL for standalone use (creates its own lr_surf).
static ANEAdam *ane_adam_compile_ex(int N, float lr, float beta1, float beta2, float eps, float wd,
                                     IOSurfaceRef shared_lr_surf) {
    if (N > ANE_ADAM_MAX)
        return _ane_adam_compile_cpu(N, lr, beta1, beta2, eps, wd);

    size_t sn = (size_t)N * 2; if (sn < 2048) sn = 2048;
    size_t ins2[2] = {sn, sn};
    size_t ins4[4] = {sn, sn, sn, sn};  // W, mn, vn, lr_in (all [1,N])

    ANEKernel *k_m = ane_compile([mil_gen_adam_m(N, beta1) dataUsingEncoding:NSUTF8StringEncoding],
                                  nil, 2, ins2, 1, &sn);
    ANEKernel *k_v = ane_compile([mil_gen_adam_v(N, beta2) dataUsingEncoding:NSUTF8StringEncoding],
                                  nil, 2, ins2, 1, &sn);
    ANEKernel *k_w = ane_compile(
        [mil_gen_adam_w(N, fmaxf(eps, 1e-4f), wd) dataUsingEncoding:NSUTF8StringEncoding],
        nil, 4, ins4, 1, &sn);

    if (!k_m || !k_v || !k_w) {
        fprintf(stderr, "ane_adam_compile FAILED (N=%d)\n", N);
        return _ane_adam_compile_cpu(N, lr, beta1, beta2, eps, wd);
    }

    ANEAdam *opt = (ANEAdam *)calloc(1, sizeof(ANEAdam));
    opt->N = N; opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps; opt->wd = wd;
    opt->k_m = k_m; opt->k_v = k_v; opt->k_w = k_w;

    opt->dw_surf = k_m->ioInputs[0];
    opt->m_surf  = k_m->ioInputs[1];

    // Wire dw into k_v slot 0
    IOSurfaceRef kv_ins[2] = {opt->dw_surf, NULL};
    ane_rewire(k_v, kv_ins, NULL);
    opt->v_surf = k_v->ioInputs[1];

    // ANE slot ordering (determined empirically): [0]=W, [1]=mn, [2]=lr_in, [3]=vn
    IOSurfaceRef kw_ins[4] = {NULL, k_m->ioOutputs[0], NULL, k_v->ioOutputs[0]};
    ane_rewire(k_w, kw_ins, NULL);
    opt->w_surf     = k_w->ioInputs[0];
    opt->w_new_surf = k_w->ioOutputs[0];
    opt->lr_surf    = k_w->ioInputs[2];  // slot 2 = lr_in
    (void)shared_lr_surf;  // shared_lr_surf no longer used — each opt has own lr_surf

    // Zero-init m and v
    IOSurfaceLock(opt->m_surf, 0, NULL);
    memset(IOSurfaceGetBaseAddress(opt->m_surf), 0, sn);
    IOSurfaceUnlock(opt->m_surf, 0, NULL);
    IOSurfaceLock(opt->v_surf, 0, NULL);
    memset(IOSurfaceGetBaseAddress(opt->v_surf), 0, sn);
    IOSurfaceUnlock(opt->v_surf, 0, NULL);

    return opt;
}

static ANEAdam *ane_adam_compile(int N, float lr, float beta1, float beta2, float eps, float wd) {
    return ane_adam_compile_ex(N, lr, beta1, beta2, eps, wd, NULL);
}

// Update the learning rate for this optimizer (used on next ane_adam_step call).
static void ane_adam_set_lr(ANEAdam *opt, float lr) { opt->lr = lr; }
static void ane_adam_set_clip(ANEAdam *opt, float clip) { opt->dw_clip = clip; }

// Write bias-corrected lr_t to an optimizer's lr_surf (fills all N fp16 slots).
// lr_in is now [1,N] — same shape as W/m/v — so every element must equal lr_t.
static void ane_adam_write_lr(IOSurfaceRef lr_surf, float lr, float beta1, float beta2, int t) {
    float b1t = powf(beta1, (float)t);
    float b2t = powf(beta2, (float)t);
    float lr_t = lr * sqrtf(1.0f - b2t) / (1.0f - b1t);
    // fp16 min normal = 2^-14 ≈ 6.1e-5. ANE FTZ flushes subnormals to 0.
    const float fp16_min_normal = 6.103515625e-5f;
    if (lr_t > 0.f && lr_t < fp16_min_normal) lr_t = fp16_min_normal;
    _Float16 lr_fp16 = (_Float16)lr_t;
    IOSurfaceLock(lr_surf, 0, NULL);
    _Float16 *base = (_Float16 *)IOSurfaceGetBaseAddress(lr_surf);
    size_t n = IOSurfaceGetAllocSize(lr_surf) / 2;
    float16x8_t vfill = vdupq_n_f16((__fp16)lr_fp16);
    size_t i = 0;
    for (; i + 7 < n; i += 8) vst1q_f16((__fp16*)(base+i), vfill);
    for (; i < n; i++) base[i] = lr_fp16;
    IOSurfaceUnlock(lr_surf, 0, NULL);
}

// Write lr to a single optimizer (ANE path only; CPU path uses opt->lr directly).
static void ane_adam_set_lr_surf(ANEAdam *opt, float lr, float beta1, float beta2, int t) {
    if (opt->cpu_mode) { opt->lr = lr; return; }
    ane_adam_write_lr(opt->lr_surf, lr, beta1, beta2, t);
    opt->lr = lr;
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
    IOSurfaceRef ins[4] = {w_surf, NULL, NULL, NULL};  // slot 3 = lr_surf (NULL = keep existing)
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
    float owd = 1.0f - opt->wd;
    float32x4_t vlr=vdupq_n_f32(lr_t), veps=vdupq_n_f32(eps), vowd=vdupq_n_f32(owd);

    float clip = opt->dw_clip;
    int i = 0;
    for (; i <= N-4; i += 4) {
        float32x4_t dw = vcvt_f32_f16(vld1_f16((const __fp16 *)(dw_ptr+i)));
        if (clip > 0) { float32x4_t vc=vdupq_n_f32(clip); dw=vminq_f32(vmaxq_f32(dw,vnegq_f32(vc)),vc); }
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
        float32x4_t wn = vsubq_f32(vmulq_f32(vowd, w), vmulq_f32(vlr, vmulq_f32(mi, rd)));
        vst1_f16((__fp16 *)(wn_ptr+i), vcvt_f16_f32(wn));
    }
    for (; i < N; i++) {
        float dw=(float)dw_ptr[i];
        if (clip > 0) { if (dw > clip) dw=clip; else if (dw < -clip) dw=-clip; }
        float w=(float)w_ptr[i];
        m[i] = b1*m[i] + ob1*dw;
        v[i] = b2*v[i] + ob2*dw*dw;
        wn_ptr[i] = (_Float16)(owd*w - lr_t*m[i]/(sqrtf(v[i])+eps));
    }

    IOSurfaceUnlock(opt->w_new_surf, 0, NULL);
    IOSurfaceUnlock(opt->w_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(opt->dw_surf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_adam_step(ANEAdam *opt, int t) {
    if (opt->cpu_mode) {
        _ane_adam_step_cpu(opt, t);
        return;
    }

    // lr_surf is written externally via ane_adam_write_lr / ane_unet_large_set_lr.
    // Do not write here — would overwrite the scheduled lr with opt->lr (base value).

    // Clip dW in-place before ANE kernels read dw_surf
    if (opt->dw_clip > 0) {
        float clip = opt->dw_clip;
        IOSurfaceLock(opt->dw_surf, 0, NULL);
        _Float16 *dw = (_Float16 *)IOSurfaceGetBaseAddress(opt->dw_surf);
        float16x8_t vc = vdupq_n_f16((__fp16)clip), vnc = vdupq_n_f16((__fp16)-clip);
        int i = 0;
        for (; i <= opt->N - 8; i += 8) {
            float16x8_t v = vld1q_f16((const __fp16 *)(dw+i));
            vst1q_f16((__fp16 *)(dw+i), vminq_f16(vmaxq_f16(v, vnc), vc));
        }
        for (; i < opt->N; i++) {
            float v = (float)dw[i];
            if (v > clip) v = clip; else if (v < -clip) v = -clip;
            dw[i] = (_Float16)v;
        }
        IOSurfaceUnlock(opt->dw_surf, 0, NULL);
    }

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
