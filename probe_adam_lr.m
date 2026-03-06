// probe_adam_lr.m — test ANE Adam k_w parameter ordering
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -o probe_adam_lr probe_adam_lr.m
#import <Foundation/Foundation.h>
#include "ane_runtime.h"
#include "mil_gen.h"

int main(void) { @autoreleasepool {
    ane_init();
    const int N = 96;
    size_t sn = 2048;
    size_t ins4[4] = {sn, sn, sn, sn};

    ANEKernel *k_w = ane_compile(
        [mil_gen_adam_w(N, 1e-4f) dataUsingEncoding:NSUTF8StringEncoding],
        nil, 4, ins4, 1, &sn);
    if (!k_w) { printf("compile failed\n"); return 1; }

    printf("k_w nInputs=%d nOutputs=%d\n", k_w->nInputs, k_w->nOutputs);

    // MIL: func main(W, mn, vn, lr_in)
    // W_new = W - lr_in * mn/sqrt(vn + eps)
    // We'll write distinctive values to each slot and observe output.
    //
    // Test: W=10, mn=1, vn=1, lr=2 → Wn = 10 - 2*1/sqrt(1+1e-4) ≈ 10 - 2 = 8
    // Vary one input at a time, fix others, see which slot controls lr behavior.

    // Helper: fill surface with fp16 scalar
    void (^fill)(IOSurfaceRef, float) = ^(IOSurfaceRef s, float v) {
        _Float16 fv = (_Float16)v;
        IOSurfaceLock(s, 0, NULL);
        uint8_t *b = (uint8_t*)IOSurfaceGetBaseAddress(s);
        for (size_t i = 0; i+1 < IOSurfaceGetAllocSize(s); i+=2) memcpy(b+i,&fv,2);
        IOSurfaceUnlock(s, 0, NULL);
    };
    float (^read3)(IOSurfaceRef) = ^(IOSurfaceRef s) {
        _Float16 v; IOSurfaceLock(s,kIOSurfaceLockReadOnly,NULL);
        memcpy(&v,(uint8_t*)IOSurfaceGetBaseAddress(s)+3*2,2);
        IOSurfaceUnlock(s,kIOSurfaceLockReadOnly,NULL);
        return (float)v;
    };

    // Baseline: all=1 except try to set W=10, lr=2 in each slot
    // Expected Wn[3] = W[3] - lr*mn[3]/sqrt(vn[3]+eps)
    // = 10 - 2*1/sqrt(1+1e-4) ≈ 8.0
    printf("\nTest: identify which slot is W, mn, vn, lr_in\n");
    printf("For each candidate slot assignment, Wn[3] should = W-lr*m/sqrt(v+eps)\n\n");

    for (int w_slot=0; w_slot<4; w_slot++) {
        for (int lr_slot=0; lr_slot<4; lr_slot++) {
            if (lr_slot == w_slot) continue;
            // Set W=10 in w_slot, lr=2 in lr_slot, all others=1
            for (int s=0; s<4; s++) fill(k_w->ioInputs[s], 1.0f);
            fill(k_w->ioInputs[w_slot], 10.0f);
            fill(k_w->ioInputs[lr_slot], 2.0f);
            ane_eval(k_w);
            float wn = read3(k_w->ioOutputs[0]);
            // Expected: 10 - 2*1/sqrt(1+1e-4) ≈ 8.0
            int match = fabsf(wn - 8.0f) < 0.1f;
            if (match)
                printf("  *** MATCH: ioInputs[%d]=W, ioInputs[%d]=lr → Wn[3]=%.4f (expected~8) ***\n", w_slot, lr_slot, wn);
        }
    }

    // Also test: find mn slot — mn=0 → Wn=W
    printf("\nTest: mn=0 → Wn should equal W (no update)\n");
    for (int mn_slot=0; mn_slot<4; mn_slot++) {
        for (int s=0; s<4; s++) fill(k_w->ioInputs[s], 1.0f);
        fill(k_w->ioInputs[mn_slot], 0.0f);
        ane_eval(k_w);
        float wn = read3(k_w->ioOutputs[0]);
        printf("  mn_slot=%d → Wn[3]=%.4f %s\n", mn_slot, wn, fabsf(wn-1.0f)<0.01f ? "(Wn=W=1 ✓)" : "");
    }

    return 0;
}}
// Additional test appended — compile fresh
