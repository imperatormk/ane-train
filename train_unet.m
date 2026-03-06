// train_unet.m — ~20M param ConvNeXt UNet training loop (ANE)
//
// Architecture: dims=[96,192,384], depths=[4,6,10,4,4], 6× global attn
// Uses first sample only (overfit test) to validate forward/backward correctness.
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework AppKit -framework IOSurface -framework CoreML -framework CoreFoundation -o train_unet train_unet.m
//
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#include <math.h>
#include <time.h>
#include "modules/blocks/ane_unet_large.h"
#include "data_utils.h"

int g_fp16_io = 1;

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e3 + ts.tv_nsec*1e-6;
}

int main(void) {
    @autoreleasepool {

    const char *data_root = getenv("DATA_ROOT");
    if (!data_root) { printf("Set DATA_ROOT env var (e.g. export DATA_ROOT=path/to/train)\n"); return 1; }
    const int H = 256;
    // UNet architecture: depths=[4,6,10,4,4], 6× global attn
    const int nB1 = 4, nB2 = 6, nB3 = 10, nB4 = 4, nB5 = 4, nA = 6;
    const int NSTEPS = 200;
    const float LR = 1e-4f;
    const int LOG_EVERY = 10;

    int S0 = H * H;

    // ---- Data ----
    int npairs = 0;
    ImgPair *pairs = collect_pairs(data_root, &npairs);
    printf("Found %d image pairs\n", npairs);
    if (npairs == 0) { printf("No pairs! Check data_root.\n"); return 1; }

    float *rgb_f32   = malloc(3 * H * H * sizeof(float));
    float *mask_f32 = malloc(1 * H * H * sizeof(float));
    _Float16 *rgb_f16   = malloc(3 * H * H * sizeof(_Float16));
    _Float16 *mask_f16 = malloc(1 * H * H * sizeof(_Float16));
    _Float16 *pred_f16  = malloc(S0 * sizeof(_Float16));

    // Load sample 0 once (overfit test)
    printf("Loading sample 0: %s\n", pairs[0].img);
    if (!load_png(pairs[0].img, 3, H, H, rgb_f32)) return 1;
    if (!load_png(pairs[0].mask, 1, H, H, mask_f32)) return 1;
    f32_to_f16(rgb_f32,   rgb_f16,   3 * H * H);
    f32_to_f16(mask_f32, mask_f16, 1 * H * H);
    { float sr=0, sa=0; for (int i=0; i<H*H; i++) { sr+=(float)rgb_f16[i]; sa+=(float)mask_f16[i]; }
      printf("  rgb mean=%.3f  mask mean=%.3f\n", sr/(H*H), sa/(H*H)); }

    // ---- Compile UNet ----
    printf("\nCompiling Large UNet H=%d nB1=%d nB2=%d nB3=%d nB4=%d nB5=%d nA=%d...\n",
           H, nB1, nB2, nB3, nB4, nB5, nA);
    ane_init();
    ane_enable_cache();
    double t0 = now_ms();
    ANEUNetLarge *net = ane_unet_large_compile(H, nB1, nB2, nB3, nB4, nB5, nA);
    if (!net) { printf("UNetLarge compile FAILED\n"); return 1; }
    printf("  fwd compile: %.0f ms\n", now_ms()-t0);

    ANEUNetLargeWeights w = ane_unet_large_make_weights(nB1, nB2, nB3, nB4, nB5, nA);
    ane_unet_large_set_weights(net, &w);

    t0 = now_ms();
    ANEUNetLargeBwd *bwd = ane_unet_large_bwd_compile(net, LR, 0.9f, 0.999f, 1e-8f, /*checkpointed=*/0);
    if (!bwd) { printf("UNetLargeBwd compile FAILED\n"); return 1; }
    printf("  bwd compile: %.0f ms\n", now_ms()-t0);

    // ---- Training loop ----
    printf("\nTraining %d steps (overfit on sample 0, lr=%.1e)...\n", NSTEPS, LR);
    printf("%-6s  %-10s  %-8s  %-8s  %-8s  %-8s\n", "step", "loss", "fwd_ms", "save_ms", "bwd_ms", "total_ms");

    for (int step = 1; step <= NSTEPS; step++) {
        double t0 = now_ms();

        ane_unet_large_eval(net, rgb_f16);
        double t1 = now_ms();

        ane_unet_large_read_output(net, pred_f16);
        ane_unet_large_bwd_save_fwd(bwd, net, H);
        double t2 = now_ms();

        float loss = ane_unet_large_bwd_eval(bwd, net, pred_f16, mask_f16, step);
        double t3 = now_ms();

        if ((step >= 2 && step <= 20) || step % LOG_EVERY == 0) {
            printf("%-6d  %-10.4f  %-8.1f  %-8.1f  %-8.1f  %-8.1f\n",
                   step, loss, t1-t0, t2-t1, t3-t2, t3-t0);
        }
    }

    printf("\nDone.\n");
    free(rgb_f32); free(mask_f32);
    free(rgb_f16); free(mask_f16); free(pred_f16);
    free(pairs);
    return 0;
    }
}
