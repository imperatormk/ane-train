// ane_upsample2x.h — Nearest-2x upsample (CPU NEON)
//
// ANE cannot do this: scatter matrix P is always larger than X, violating slot0≤slot1 rule.
// CPU nearest-2x is essentially 4 memcpy patterns — negligible cost.
//
// Input:  x[C, H, H] fp16
// Output: y[C, 2H, 2H] fp16
//
// Usage:
//   _Float16 y[C * 4*H*H];
//   ane_upsample2x(x, y, C, H);
//
#pragma once
#include <stdint.h>
#include <string.h>
#include <arm_neon.h>

// x[C, H, H] fp16 → y[C, 2H, 2H] fp16, nearest-neighbor 2x
static void ane_upsample2x(const _Float16 *x, _Float16 *y, int C, int H) {
    int H2 = H * 2;
    for (int c = 0; c < C; c++) {
        const _Float16 *xc = x + c * H * H;
        _Float16 *yc = y + c * H2 * H2;
        for (int h = 0; h < H; h++) {
            const _Float16 *xrow = xc + h * H;
            _Float16 *yr0 = yc + (h*2)   * H2;
            _Float16 *yr1 = yc + (h*2+1) * H2;
            // Each pixel → 2 consecutive output pixels in row
            int w = 0;
            for (; w + 3 < H; w += 4) {
                // Load 4 fp16 pixels
                float16x4_t v = vld1_f16((const __fp16 *)(xrow + w));
                // Interleave: 1 1 2 2 3 3 4 4
                float16x8_t vv = vzipq_f16(vcombine_f16(v, v), vcombine_f16(v, v)).val[0];
                // Actually zip produces interleaved pairs — use vzip directly
                float16x4x2_t z = vzip_f16(v, v);
                float16x8_t row_out = vcombine_f16(z.val[0], z.val[1]);
                vst1q_f16((__fp16 *)(yr0 + w*2), row_out);
                vst1q_f16((__fp16 *)(yr1 + w*2), row_out);
            }
            for (; w < H; w++) {
                _Float16 v = xrow[w];
                yr0[w*2] = yr0[w*2+1] = v;
                yr1[w*2] = yr1[w*2+1] = v;
            }
        }
    }
}
