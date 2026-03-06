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
