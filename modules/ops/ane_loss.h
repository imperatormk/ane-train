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

// L1 + BCE combined (equal weight 0.5 each).
// L1 for absolute accuracy, BCE for boundary sharpness.
// NEON: abs/sign via vabsq, vcltq; log remains scalar (no NEON log).
static float l1bce_loss_and_grad(const _Float16 *p, const _Float16 *t,
                                  _Float16 *grad, int S) {
    float eps = 1e-7f, inv_S = 1.0f / (float)S;
    float loss_l1 = 0.0f, loss_bce = 0.0f;

    // Scalar loop — log has no NEON equivalent, bottleneck is logf calls.
    // Vectorize the grad write (4-wide fp32 → fp16) to reduce store overhead.
    int i = 0;
    for (; i <= S - 4; i += 4) {
        float p0=(float)p[i],   t0=(float)t[i];
        float p1=(float)p[i+1], t1=(float)t[i+1];
        float p2=(float)p[i+2], t2=(float)t[i+2];
        float p3=(float)p[i+3], t3=(float)t[i+3];

        float d0=p0-t0, d1=p1-t1, d2=p2-t2, d3=p3-t3;
        loss_l1  += (d0<0?-d0:d0) + (d1<0?-d1:d1) + (d2<0?-d2:d2) + (d3<0?-d3:d3);
        loss_bce += -(t0*logf(p0+eps)+(1-t0)*logf(1-p0+eps))
                  + -(t1*logf(p1+eps)+(1-t1)*logf(1-p1+eps))
                  + -(t2*logf(p2+eps)+(1-t2)*logf(1-p2+eps))
                  + -(t3*logf(p3+eps)+(1-t3)*logf(1-p3+eps));

        float32x4_t vg_l1 = {
            (d0>0?1.f:d0<0?-1.f:0.f),
            (d1>0?1.f:d1<0?-1.f:0.f),
            (d2>0?1.f:d2<0?-1.f:0.f),
            (d3>0?1.f:d3<0?-1.f:0.f),
        };
        float32x4_t vg_bce = {
            -t0/(p0+eps) + (1-t0)/(1-p0+eps),
            -t1/(p1+eps) + (1-t1)/(1-p1+eps),
            -t2/(p2+eps) + (1-t2)/(1-p2+eps),
            -t3/(p3+eps) + (1-t3)/(1-p3+eps),
        };
        float32x4_t vinvS = vdupq_n_f32(inv_S);
        float32x4_t vg = vmulq_f32(vaddq_f32(vmulq_n_f32(vg_l1, 0.5f),
                                              vmulq_n_f32(vg_bce, 0.5f)), vinvS);
        vst1_f16((__fp16*)(grad+i), vcvt_f16_f32(vg));
    }
    for (; i < S; i++) {
        float pi=(float)p[i], ti=(float)t[i], d=pi-ti;
        loss_l1  += d < 0.f ? -d : d;
        loss_bce += -(ti*logf(pi+eps) + (1-ti)*logf(1-pi+eps));
        float g = 0.5f*((d>0?1.f:d<0?-1.f:0.f) + (-ti/(pi+eps)+(1-ti)/(1-pi+eps))) * inv_S;
        grad[i] = (_Float16)g;
    }
    return 0.5f * loss_l1 * inv_S + 0.5f * loss_bce * inv_S;
}
