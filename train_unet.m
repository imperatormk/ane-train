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

// Save side-by-side tile: [GT | Pred], each H×H grayscale → H×(2H) wide PNG
static void save_pred_png(const _Float16 *pred, const _Float16 *gt, int H,
                          int step, NSString *run_dir) {
    int S = H * H;
    int W = 2 * H;
    uint8_t *buf = malloc(H * W);
    // Left: GT
    for (int i = 0; i < S; i++) {
        float v = (float)gt[i];
        if (v < 0) v = 0; if (v > 1) v = 1;
        int row = i / H, col = i % H;
        buf[row * W + col] = (uint8_t)(v * 255.0f);
    }
    // Right: Pred
    for (int i = 0; i < S; i++) {
        float v = (float)pred[i];
        if (v < 0) v = 0; if (v > 1) v = 1;
        int row = i / H, col = i % H;
        buf[row * W + H + col] = (uint8_t)(v * 255.0f);
    }
    NSBitmapImageRep *rep = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL pixelsWide:W pixelsHigh:H
        bitsPerSample:8 samplesPerPixel:1 hasAlpha:NO isPlanar:NO
        colorSpaceName:NSDeviceWhiteColorSpace bytesPerRow:W bitsPerPixel:8];
    memcpy(rep.bitmapData, buf, H * W);
    free(buf);
    NSData *png = [rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
    NSString *path = [run_dir stringByAppendingPathComponent:
                      [NSString stringWithFormat:@"pred_%04d.png", step]];
    [png writeToFile:path atomically:YES];
}

int main(void) {
    @autoreleasepool {

    const char *data_root = getenv("DATA_ROOT");
    if (!data_root) { printf("Set DATA_ROOT env var (e.g. export DATA_ROOT=path/to/train)\n"); return 1; }
    const int H = 256;
    // UNet architecture: depths=[4,6,10,4,4], 6× global attn
    const int nB1 = 4, nB2 = 6, nB3 = 10, nB4 = 4, nB5 = 4, nA = 6;
    const int NEPOCHS = 20;
    const float LR_MAX = 1e-4f;
    const float LR_MIN = 1e-5f;
    const int WARMUP_EPOCHS = 3;  // linear warmup over first 3 epochs (like PyTorch)
    const int LOG_EVERY = 20;

    const float loss_scale = 512.0f;

    int S0 = H * H;

    // ---- Data ----
    int npairs = 0;
    ImgPair *pairs = collect_pairs(data_root, &npairs);
    printf("Found %d image pairs\n", npairs);
    if (npairs == 0) { printf("No pairs! Check data_root.\n"); return 1; }

    float *rgb_f32   = malloc(3 * H * H * sizeof(float));
    float *mask_f32  = malloc(1 * H * H * sizeof(float));
    _Float16 *rgb_f16   = malloc(3 * H * H * sizeof(_Float16));
    _Float16 *mask_f16  = malloc(1 * H * H * sizeof(_Float16));
    _Float16 *pred_f16  = malloc(S0 * sizeof(_Float16));
    printf("Training on %d pairs\n", npairs);

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
    ANEUNetLargeBwd *bwd = ane_unet_large_bwd_compile(net, LR_MAX, 0.9f, 0.999f, 1e-8f, /*wd=*/0.001f, /*checkpointed=*/0);
    if (!bwd) { printf("UNetLargeBwd compile FAILED\n"); return 1; }
    printf("  bwd compile: %.0f ms\n", now_ms()-t0);
    ane_unet_large_set_dw_clip(bwd, 1000.0f);  // gradient clipping: cap |dW| per element

    // ---- Run dir ----
    NSDateFormatter *fmt = [[NSDateFormatter alloc] init];
    fmt.dateFormat = @"yyyyMMdd_HHmmss";
    NSString *run_dir = [NSString stringWithFormat:@"runs/%@", [fmt stringFromDate:[NSDate date]]];
    [[NSFileManager defaultManager] createDirectoryAtPath:run_dir
        withIntermediateDirectories:YES attributes:nil error:nil];
    printf("Saving predictions to %s/\n", run_dir.UTF8String);

    // ---- Shuffle indices (Fisher-Yates) ----
    int *order = malloc(npairs * sizeof(int));
    for (int i = 0; i < npairs; i++) order[i] = i;

    // ---- Training loop ----
    int warmup_steps = WARMUP_EPOCHS * npairs;
    int total_steps  = NEPOCHS * npairs;
    printf("\nTraining %d epochs × %d pairs = %d steps (lr %.1e→%.1e cosine, %d warmup steps)...\n",
           NEPOCHS, npairs, total_steps, LR_MAX, LR_MIN, warmup_steps);
    printf("%-6s  %-4s  %-10s  %-8s  %-8s  %-8s  %-8s\n",
           "step", "ep", "loss", "fwd_ms", "save_ms", "bwd_ms", "total_ms");
    printf("  (loss_scale=%.0f)\n", loss_scale);

    int step = 0;
    for (int epoch = 0; epoch < NEPOCHS; epoch++) {
        // Shuffle for this epoch — seeded by epoch for reproducibility
        srand(42 + epoch);
        for (int i = npairs - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        for (int ei = 0; ei < npairs; ei++) {
            step++;
            double t0 = now_ms();

            int idx = order[ei];
            if (!load_png(pairs[idx].img,  3, H, H, rgb_f32))  continue;
            if (!load_png(pairs[idx].mask, 1, H, H, mask_f32)) continue;
            f32_to_f16(rgb_f32,  rgb_f16,  3 * H * H);
            f32_to_f16(mask_f32, mask_f16, 1 * H * H);

            ane_unet_large_eval(net, rgb_f16);
            double t1 = now_ms();

            ane_unet_large_read_output(net, pred_f16);
            ane_unet_large_bwd_save_fwd(bwd, net, H);
            double t2 = now_ms();

            // Cosine LR with linear warmup (epoch-based, like PyTorch)
            float lr_t;
            if (step <= warmup_steps) {
                lr_t = LR_MAX * (float)step / (float)warmup_steps;
            } else {
                float progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                lr_t = LR_MIN + 0.5f * (LR_MAX - LR_MIN) * (1.0f + cosf(M_PI * progress));
            }
            ane_unet_large_set_lr(bwd, lr_t, step);

            int overflow = 0;
            float loss = ane_unet_large_bwd_eval(bwd, net, pred_f16, mask_f16, step,
                                                  loss_scale, &overflow);
            double t3 = now_ms();

            // Detect loss spike (>3× previous) for diagnostics
            static float prev_loss = 0.f;
            int spike = prev_loss > 0.f && loss > prev_loss * 3.0f;
            if (!overflow) prev_loss = loss;

            if (spike)
                ane_unet_large_print_spike_diag(bwd, net, step);

            int log_step = (epoch == 0 && ei < 20) || step % LOG_EVERY == 0 || spike || overflow;
            if (log_step) {
                printf("%-6d  %-4d  %-10.4f  %-8.1f  %-8.1f  %-8.1f  %-8.1f  lr=%.2e%s%s\n",
                       step, epoch+1, loss, t1-t0, t2-t1, t3-t2, t3-t0, lr_t,
                       overflow ? " SKIP" : "",
                       spike    ? " SPIKE" : "");
            }
            if (step % 100 == 0)
                save_pred_png(pred_f16, mask_f16, H, step, run_dir);
        }
        printf("-- epoch %d done, step %d --\n", epoch+1, step);
    }
    free(order);

    printf("\nDone.\n");
    free(rgb_f32); free(mask_f32);
    free(rgb_f16); free(mask_f16); free(pred_f16);
    free(pairs);
    return 0;
    }
}
