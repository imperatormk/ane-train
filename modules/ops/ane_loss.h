// ane_loss.h — Loss functions + gradients (CPU, fp16 in/out)
//
// All losses operate on sigmoid-activated predictions p∈(0,1), target t∈[0,1].
//
// BCE:     L = -mean( t*log(p+ε) + (1-t)*log(1-p+ε) )
// L1:      L = mean( |p - t| )
// L1+BCE:  L = 0.5*L1 + 0.5*BCE
//
#pragma once
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>

// BCE loss + gradient.
// p[S]: predicted (after sigmoid), t[S]: target, grad[S]: dL/dp output.
static float bce_loss_and_grad(const _Float16 *p, const _Float16 *t,
                                _Float16 *grad, int S) {
    float eps = 1e-7f, inv_S = 1.0f / (float)S, loss = 0.0f;
    for (int i = 0; i < S; i++) {
        float pi = (float)p[i], ti = (float)t[i];
        loss += -(ti * logf(pi + eps) + (1.0f - ti) * logf(1.0f - pi + eps));
        grad[i] = (_Float16)((-ti/(pi+eps) + (1.0f-ti)/(1.0f-pi+eps)) * inv_S);
    }
    return loss * inv_S;
}

// L1 loss + gradient.
// grad = sign(p-t) / S
static float l1_loss_and_grad(const _Float16 *p, const _Float16 *t,
                               _Float16 *grad, int S) {
    float inv_S = 1.0f / (float)S, loss = 0.0f;
    for (int i = 0; i < S; i++) {
        float d = (float)p[i] - (float)t[i];
        loss += d < 0.0f ? -d : d;
        grad[i] = (_Float16)((d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f)) * inv_S);
    }
    return loss * inv_S;
}

// ---- Dynamic loss scaling ----
// Call after loss+grad computation. Detects inf/nan in grad, halves scale.
// Returns 1 if grad is valid (use it), 0 if overflow (skip step, scale halved).
static int loss_scale_check_and_update(const _Float16 *grad, int S,
                                        float *scale, float scale_up_every,
                                        int *steps_since_overflow) {
    // Check for inf/nan
    for (int i = 0; i < S; i++) {
        float v = (float)grad[i];
        if (v != v || v > 65504.f || v < -65504.f) {
            *scale *= 0.5f;
            if (*scale < 1.f) *scale = 1.f;
            *steps_since_overflow = 0;
            return 0;  // skip this step
        }
    }
    // Scale up periodically if no overflow
    (*steps_since_overflow)++;
    if (*steps_since_overflow >= (int)scale_up_every) {
        *scale *= 2.f;
        if (*scale > 65536.f) *scale = 65536.f;
        *steps_since_overflow = 0;
    }
    return 1;
}

// Apply loss scale to gradient in-place (fp16, NEON).
static void loss_scale_apply(const float scale, _Float16 *grad, int S) {
    float32x4_t vs = vdupq_n_f32(scale);
    int i = 0;
    for (; i <= S - 4; i += 4) {
        float32x4_t v = vcvt_f32_f16(vld1_f16((__fp16*)(grad+i)));
        vst1_f16((__fp16*)(grad+i), vcvt_f16_f32(vmulq_f32(v, vs)));
    }
    for (; i < S; i++) grad[i] = (_Float16)((float)grad[i] * scale);
}

// Unscale gradient in-place after Adam (divide by scale).
static void loss_scale_unscale(const float scale, _Float16 *grad, int S) {
    loss_scale_apply(1.0f / scale, grad, S);
}

// L1 + BCE combined (equal weight 0.5 each).
// L1 for absolute accuracy, BCE for boundary sharpness.
// NEON: abs/sign via vabsq, vcltq; log remains scalar (no NEON log).
static float l1bce_loss_and_grad(const _Float16 *p, const _Float16 *t,
                                  _Float16 *grad, int S) {
    // Clamp p to [eps, 1-eps] before BCE — prevents -t/(p+eps) blowing up
    // when predictions saturate near 0 or 1 after sufficient training.
    float eps = 1e-7f, pmin = 1e-4f, pmax = 1.f - 1e-4f;
    float inv_S = 1.0f / (float)S;
    float grad_scale = inv_S;
    float loss_l1 = 0.0f, loss_bce = 0.0f;
#define _CLAMP(x) ((x)<pmin?pmin:((x)>pmax?pmax:(x)))

    // Scalar loop — log has no NEON equivalent, bottleneck is logf calls.
    // Vectorize the grad write (4-wide fp32 → fp16) to reduce store overhead.
    int i = 0;
    for (; i <= S - 4; i += 4) {
        float p0=(float)p[i],   t0=(float)t[i];
        float p1=(float)p[i+1], t1=(float)t[i+1];
        float p2=(float)p[i+2], t2=(float)t[i+2];
        float p3=(float)p[i+3], t3=(float)t[i+3];
        float c0=_CLAMP(p0), c1=_CLAMP(p1), c2=_CLAMP(p2), c3=_CLAMP(p3);

        float d0=p0-t0, d1=p1-t1, d2=p2-t2, d3=p3-t3;
        loss_l1  += (d0<0?-d0:d0) + (d1<0?-d1:d1) + (d2<0?-d2:d2) + (d3<0?-d3:d3);
        loss_bce += -(t0*logf(c0+eps)+(1-t0)*logf(1-c0+eps))
                  + -(t1*logf(c1+eps)+(1-t1)*logf(1-c1+eps))
                  + -(t2*logf(c2+eps)+(1-t2)*logf(1-c2+eps))
                  + -(t3*logf(c3+eps)+(1-t3)*logf(1-c3+eps));

        float32x4_t vg_l1 = {
            (d0>0?1.f:d0<0?-1.f:0.f),
            (d1>0?1.f:d1<0?-1.f:0.f),
            (d2>0?1.f:d2<0?-1.f:0.f),
            (d3>0?1.f:d3<0?-1.f:0.f),
        };
        float32x4_t vg_bce = {
            -t0/(c0+eps) + (1-t0)/(1-c0+eps),
            -t1/(c1+eps) + (1-t1)/(1-c1+eps),
            -t2/(c2+eps) + (1-t2)/(1-c2+eps),
            -t3/(c3+eps) + (1-t3)/(1-c3+eps),
        };
        float32x4_t vg = vmulq_f32(vaddq_f32(vmulq_n_f32(vg_l1, 0.5f),
                                              vmulq_n_f32(vg_bce, 0.5f)),
                                   vdupq_n_f32(grad_scale));
        vst1_f16((__fp16*)(grad+i), vcvt_f16_f32(vg));
    }
    for (; i < S; i++) {
        float pi=(float)p[i], ci=_CLAMP(pi), ti=(float)t[i], d=pi-ti;
        loss_l1  += d < 0.f ? -d : d;
        loss_bce += -(ti*logf(ci+eps) + (1-ti)*logf(1-ci+eps));
        float g = 0.5f*((d>0?1.f:d<0?-1.f:0.f) + (-ti/(ci+eps)+(1-ti)/(1-ci+eps))) * grad_scale;
        grad[i] = (_Float16)g;
    }
#undef _CLAMP
    return 0.5f * loss_l1 * inv_S + 0.5f * loss_bce * inv_S;
}
