// ane_upsample2x_bwd.h — Backward through nearest-neighbor 2× upsample (CPU NEON)
//
// Forward: each pixel copied to 2×2 block → dy[C,2H,2H]
// Backward: sum 2×2 blocks → dx[C,H,H]
//   dx[c,h,w] = dy[c,2h,2w] + dy[c,2h,2w+1] + dy[c,2h+1,2w] + dy[c,2h+1,2w+1]
//
// Layout: [C, H, H] (C-major)
//
#pragma once
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Backward through bilinear 2x upsample — gather pattern, NEON vectorized.
//
// Forward bilinear weights (align_corners=False):
//   out[2h,  2w  ] = x[h,w]                         (w=1)
//   out[2h,  2w+1] = 0.5*x[h,w] + 0.5*x[h,w+1]    (w=0.5 each)
//   out[2h+1,2w  ] = 0.5*x[h,w] + 0.5*x[h+1,w]    (w=0.5 each)
//   out[2h+1,2w+1] = 0.25*(x[h,w]+x[h,w+1]+x[h+1,w]+x[h+1,w+1])
//
// Transposed (gather): dx[h,w] = sum of dy * weight for each output that used x[h,w].
// For interior pixels (not last row/col):
//   dx[h,w] = dy[2h,2w]*1 + dy[2h,2w-1]*0.5 + dy[2h-1,2w]*0.5 + dy[2h-1,2w-1]*0.25
//           + dy[2h,2w+1]*0.5 + dy[2h+1,2w]*0.5 + dy[2h+1,2w+1]*0.25
//           + dy[2h-1,2w+1]*0.25 + dy[2h+1,2w-1]*0.25  (cross terms from neighbors)
//
// Simpler equivalent: for each input row h, the contribution from output rows 2h and 2h+1
// depends only on columns 2w-1..2w+1. We handle this row-pair by row-pair with NEON.
//
// Compact form per interior pixel:
//   dx[h,w] = 1*g[2h,2w] + 0.5*(g[2h,2w-1]+g[2h,2w+1]+g[2h-1,2w]+g[2h+1,2w])
//           + 0.25*(g[2h-1,2w-1]+g[2h-1,2w+1]+g[2h+1,2w-1]+g[2h+1,2w+1])
// Edge pixels clamp (mirror the clamping in the forward pass).
static void __attribute__((noinline))
ane_upsample2x_bilinear_bwd(const _Float16 *dy, _Float16 *dx, int C, int H) {
    int H2 = 2 * H;
    float32x4_t v025 = vdupq_n_f32(0.25f), v05 = vdupq_n_f32(0.5f);
    for (int c = 0; c < C; c++) {
        const _Float16 *dyc = dy + c * H2 * H2;
        _Float16       *dxc = dx + c * H  * H;
        for (int h = 0; h < H; h++) {
            // Output rows that contribute to input row h:
            //   primary:   r0=2h,   r1=2h+1
            //   above:     ra=2h-1 (clamp to 0)
            //   below:     rb=2h+2 (clamp to H2-1)
            int iy0 = 2*h, iy1 = 2*h+1;
            int iya = (h > 0)   ? 2*h-1 : 0;
            int iyb = (h < H-1) ? 2*h+2 : H2-1;
            const _Float16 *ra = dyc + iya * H2;
            const _Float16 *r0 = dyc + iy0 * H2;
            const _Float16 *r1 = dyc + iy1 * H2;
            const _Float16 *rb = dyc + iyb * H2;
            int w = 0;
            for (; w + 3 < H; w += 4) {
                // For each of the 4 input pixels at columns w..w+3,
                // gather the 9 contributing output pixels per input pixel.
                // Output col indices: xl=2w-1(clamped), x0=2w, xr=2w+1, xx=2w+2(next x0)
                // We process 4 input pixels → need output cols 2w-1..2w+7

                // Load even cols (x0,x0+2,x0+4,x0+6) = cols 2w,2w+2,2w+4,2w+6
                // and odd  cols (x0-1,x0+1,x0+3,x0+5) = cols 2w-1,2w+1,2w+3,2w+5
                // Use strided gather via deinterleave of 8-wide loads.
                float16x8_t r0v = vld1q_f16((const __fp16*)(r0 + 2*w));
                float16x8_t r1v = vld1q_f16((const __fp16*)(r1 + 2*w));
                float16x8_t rav = vld1q_f16((const __fp16*)(ra + 2*w));
                float16x8_t rbv = vld1q_f16((const __fp16*)(rb + 2*w));

                // Deinterleave: even=cols 2w,2w+2,2w+4,2w+6  odd=cols 2w+1,2w+3,2w+5,2w+7
                float16x4_t r0e = vuzp1_f16(vget_low_f16(r0v), vget_high_f16(r0v));  // even
                float16x4_t r0o = vuzp2_f16(vget_low_f16(r0v), vget_high_f16(r0v));  // odd
                float16x4_t r1e = vuzp1_f16(vget_low_f16(r1v), vget_high_f16(r1v));
                float16x4_t r1o = vuzp2_f16(vget_low_f16(r1v), vget_high_f16(r1v));
                float16x4_t rae = vuzp1_f16(vget_low_f16(rav), vget_high_f16(rav));
                float16x4_t rao = vuzp2_f16(vget_low_f16(rav), vget_high_f16(rav));
                float16x4_t rbe = vuzp1_f16(vget_low_f16(rbv), vget_high_f16(rbv));
                float16x4_t rbo = vuzp2_f16(vget_low_f16(rbv), vget_high_f16(rbv));

                // Left neighbors (col 2w-1): shift r*o right by 1, pad with edge
                // For w>0 we can load from 2w-1; for w==0 clamp col -1→0 = same as col 0
                float16x4_t r0l, r1l, ral, rbl;
                if (w > 0) {
                    r0l = vld1_lane_f16((const __fp16*)(r0+2*w-1), vext_f16(vdup_n_f16(0),r0o,3), 0);
                    r1l = vld1_lane_f16((const __fp16*)(r1+2*w-1), vext_f16(vdup_n_f16(0),r1o,3), 0);
                    ral = vld1_lane_f16((const __fp16*)(ra+2*w-1), vext_f16(vdup_n_f16(0),rao,3), 0);
                    rbl = vld1_lane_f16((const __fp16*)(rb+2*w-1), vext_f16(vdup_n_f16(0),rbo,3), 0);
                } else {
                    r0l = r0e; r1l = r1e; ral = rae; rbl = rbe; // clamp: col -1 → col 0
                }

                // Convert to fp32 for accumulation
                float32x4_t g00 = vcvt_f32_f16(r0e);   // dy[2h,   2w+0,2,4,6]  weight=1
                float32x4_t g0r = vcvt_f32_f16(r0o);   // dy[2h,   2w+1,3,5,7]  weight=0.5
                float32x4_t g0l = vcvt_f32_f16(r0l);   // dy[2h,   2w-1,1,3,5]  weight=0.5
                float32x4_t g10 = vcvt_f32_f16(r1e);   // dy[2h+1, 2w]          weight=0.5
                float32x4_t g1r = vcvt_f32_f16(r1o);   // dy[2h+1, 2w+1]        weight=0.25
                float32x4_t g1l = vcvt_f32_f16(r1l);   // dy[2h+1, 2w-1]        weight=0.25
                float32x4_t ga0 = vcvt_f32_f16(rae);   // dy[2h-1, 2w]          weight=0.5
                float32x4_t gar = vcvt_f32_f16(rao);   // dy[2h-1, 2w+1]        weight=0.25
                float32x4_t gal = vcvt_f32_f16(ral);   // dy[2h-1, 2w-1]        weight=0.25
                float32x4_t gb0 = vcvt_f32_f16(rbe);   // dy[2h+2, 2w]          weight=0.5 (from h+1's top)
                float32x4_t gbr = vcvt_f32_f16(rbo);   // dy[2h+2, 2w+1]        weight=0.25
                float32x4_t gbl = vcvt_f32_f16(rbl);   // dy[2h+2, 2w-1]        weight=0.25

                float32x4_t acc = g00;
                acc = vmlaq_f32(acc, vaddq_f32(vaddq_f32(g0r,g0l), vaddq_f32(ga0,gb0)), v05);
                acc = vmlaq_f32(acc, vaddq_f32(vaddq_f32(vaddq_f32(g1r,g1l),vaddq_f32(gar,gbl)),vaddq_f32(gbr,gal)), v025);
                // Note: g10 contributes weight=0.5 as "above" for input row h+1,
                // and weight=0.5 as "below" for input row h (already covered via gb0 of row h).
                // g10 is the primary below-row contribution (rb for current h).
                acc = vmlaq_f32(acc, g10, v05);

                vst1_f16((__fp16*)(dxc + h*H + w), vcvt_f16_f32(acc));
            }
            // Scalar tail
            for (; w < H; w++) {
                int x0=2*w, xl=(w>0)?2*w-1:0, xr=(w<H-1)?2*w+1:H2-1, xb=(w<H-1)?2*w+2:H2-1;
                float acc = (float)r0[x0]
                    + 0.5f*((float)r0[xl]+(float)r0[xr]+(float)ra[x0]+(float)rb[x0])
                    + 0.5f*(float)r1[x0]
                    + 0.25f*((float)r1[xl]+(float)r1[xr]+(float)ra[xl]+(float)ra[xr]
                            +(float)rb[xl]+(float)rb[xr]);
                dxc[h*H+w] = (_Float16)acc;
            }
        }
    }
}

// dy[C, H2, H2] where H2=2*H → dx[C, H, H]
static void __attribute__((noinline))
ane_upsample2x_bwd(const _Float16 *dy, _Float16 *dx, int C, int H) {
    int H2 = 2 * H;
    for (int c = 0; c < C; c++) {
        const _Float16 *dyc = dy + c * H2 * H2;
        _Float16       *dxc = dx + c * H  * H;
        for (int h = 0; h < H; h++) {
            const _Float16 *row0 = dyc + (2*h)   * H2;
            const _Float16 *row1 = dyc + (2*h+1) * H2;
            int w = 0;
            for (; w + 3 < H; w += 4) {
                // Load 4 pairs from each row: row0[2w..2w+7], row1[2w..2w+7]
                float16x8_t r0 = vld1q_f16((const __fp16*)(row0 + 2*w));
                float16x8_t r1 = vld1q_f16((const __fp16*)(row1 + 2*w));
                // Sum rows: r01[i] = r0[i] + r1[i]
                float16x8_t r01 = vaddq_f16(r0, r1);
                // Pairwise add: even + odd indices
                // vpaddq_f16 adds adjacent pairs: [a0+a1, a2+a3, a4+a5, a6+a7]
                float16x4_t lo = vpadd_f16(vget_low_f16(r01), vget_high_f16(r01));
                vst1_f16((__fp16*)(dxc + h*H + w), lo);
            }
            for (; w < H; w++) {
                float s = (float)row0[2*w] + (float)row0[2*w+1]
                        + (float)row1[2*w] + (float)row1[2*w+1];
                dxc[h*H + w] = (_Float16)s;
            }
        }
    }
}
