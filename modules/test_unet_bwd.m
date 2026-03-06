// test_unet_bwd.m — Full UNetLarge backward smoke test
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML -framework IOSurface -framework CoreFoundation -o modules/test_unet_bwd modules/test_unet_bwd.m
#import <Foundation/Foundation.h>
#include <math.h>
#include <time.h>
#include "blocks/ane_unet_large.h"

int g_fp16_io = 1;

static int g_pass = 0, g_fail = 0;

#define CHECK(label, cond) do { \
    if (cond) { printf("  PASS: %s\n", label); g_pass++; } \
    else       { printf("  FAIL: %s\n", label); g_fail++; } \
} while(0)

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e3 + ts.tv_nsec*1e-6;
}

static void test_unet_large_bwd(void) {
    printf("\n[UNetLarge bwd H=256 nB1=1 nB2=1 nB3=1 nB4=1 nB5=1 nA=1]\n");
    // Use minimal depth (1 block per stage, 1 attn) for fast compile
    int H=256, nB1=1, nB2=1, nB3=1, nB4=1, nB5=1, nA=1;
    int S0 = H*H;

    ane_init();
    ane_enable_cache();

    double t0 = now_ms();
    ANEUNetLarge *net = ane_unet_large_compile(H, nB1, nB2, nB3, nB4, nB5, nA);
    printf("  fwd compile: %.0f ms\n", now_ms()-t0);
    CHECK("fwd compile", net != NULL);
    if (!net) return;

    ANEUNetLargeWeights w = ane_unet_large_make_weights(nB1, nB2, nB3, nB4, nB5, nA);
    ane_unet_large_set_weights(net, &w);

    t0 = now_ms();
    ANEUNetLargeBwd *bwd = ane_unet_large_bwd_compile(net, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.001f, /*checkpointed=*/0);
    printf("  bwd compile: %.0f ms\n", now_ms()-t0);
    CHECK("bwd compile", bwd != NULL);
    if (!bwd) return;

    _Float16 *x      = (_Float16 *)malloc(3 * S0 * 2);
    _Float16 *pred   = (_Float16 *)malloc(S0 * 2);
    _Float16 *target = (_Float16 *)malloc(S0 * 2);
    srand(42);
    for (int i = 0; i < 3*S0; i++) x[i]      = (_Float16)(0.5f + 0.01f*(i%17 - 8));
    for (int i = 0; i < S0;   i++) target[i]  = (_Float16)(i % 2);

    // Forward pass
    t0 = now_ms();
    ane_unet_large_eval(net, x);
    printf("  fwd: %.1f ms\n", now_ms()-t0);

    ane_unet_large_read_output(net, pred);
    int nan_p = 0;
    for (int i = 0; i < S0; i++) if (pred[i] != pred[i]) nan_p = 1;
    CHECK("fwd output no NaN", !nan_p);

    // Save activations for backward
    ane_unet_large_bwd_save_fwd(bwd, net, H);

    // Set lr for step 1
    ane_unet_large_set_lr(bwd, 1e-4f, 1);

    // Backward pass
    t0 = now_ms();
    int overflow = 0;
    float loss = ane_unet_large_bwd_eval(bwd, net, pred, target, 1, 128.0f, &overflow);
    printf("  bwd: %.1f ms\n", now_ms()-t0);

    CHECK("loss finite", isfinite(loss));
    CHECK("loss > 0", loss > 0.0f);
    CHECK("no overflow", !overflow);
    printf("  loss = %.4f\n", loss);

    // dx_stem should be non-NaN
    int nan_dx = 0;
    for (int i = 0; i < 3*S0; i++) if (bwd->dx_stem[i] != bwd->dx_stem[i]) nan_dx = 1;
    CHECK("dx_stem no NaN", !nan_dx);

    // Step 2: fwd + bwd should still work after weight update
    ane_unet_large_eval(net, x);
    ane_unet_large_read_output(net, pred);
    ane_unet_large_bwd_save_fwd(bwd, net, H);
    ane_unet_large_set_lr(bwd, 1e-4f, 2);
    float loss2 = ane_unet_large_bwd_eval(bwd, net, pred, target, 2, 128.0f, &overflow);
    CHECK("step 2 loss finite", isfinite(loss2));
    printf("  loss2 = %.4f\n", loss2);

    // Loss should decrease (or at least not explode) after one step
    CHECK("loss not exploding", loss2 < loss * 5.0f);

    free(x); free(pred); free(target);
}

int main(void) {
    @autoreleasepool {
    printf("=== UNetLarge Backward Test ===\n");
    test_unet_large_bwd();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
    }
}
