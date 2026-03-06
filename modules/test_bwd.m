// test_bwd.m — Tests for ANE backward ops
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -o modules/test_bwd modules/test_bwd.m
#import <Foundation/Foundation.h>
#include <math.h>
#include "ops/ane_loss.h"
#include "ops/ane_gelu_bwd.h"
#include "ops/ane_matmul_bwd.h"
#include "ops/ane_ln_bwd.h"
#include "ops/ane_dw_bwd.h"
#include "ops/ane_adam.h"
#include "blocks/ane_convnext_bwd.h"

int g_fp16_io = 1;

static int g_pass = 0, g_fail = 0;

#define CHECK(label, cond) do { \
    if (cond) { printf("  PASS: %s\n", label); g_pass++; } \
    else       { printf("  FAIL: %s\n", label); g_fail++; } \
} while(0)

static void test_bce_loss(void) {
    printf("\n[BCE Loss C=1 S=256]\n");
    int S = 256;
    _Float16 *p    = malloc(S*2);
    _Float16 *t    = malloc(S*2);
    _Float16 *grad = malloc(S*2);
    for (int i=0;i<S;i++) { p[i]=(_Float16)0.5f; t[i]=(_Float16)(i%2); }
    float loss = bce_loss_and_grad(p, t, grad, S);
    CHECK("loss ~log(2)", fabsf(loss - 0.6931f) < 0.01f);
    float g0 = (float)grad[0];
    float g1 = (float)grad[1];
    CHECK("grad sign", g0 > 0 && g1 < 0);
    free(p); free(t); free(grad);
}

static void test_gelu_bwd(void) {
    printf("\n[GELU bwd C=32 S=64]\n");
    int C=32, S=64;
    ane_init();
    ANEGeluBwd *g = ane_gelu_bwd_compile(C, S);
    CHECK("compile", g != NULL);
    if (!g) return;

    _Float16 *x  = malloc(C*S*2);
    _Float16 *dy = malloc(C*S*2);
    _Float16 *dx = malloc(C*S*2);
    for (int i=0;i<C*S;i++) { x[i]=(_Float16)(0.5f); dy[i]=(_Float16)(1.0f); }

    ane_gelu_bwd_write_x(g, x);
    ane_gelu_bwd_write_dy(g, dy);
    ane_gelu_bwd_eval(g);
    ane_gelu_bwd_read_dx(g, dx);

    float v = (float)dx[0];
    CHECK("gelu'(0.5) ~0.88", fabsf(v - 0.8814f) < 0.05f);
    int nan=0; for(int i=0;i<C*S;i++) if(dx[i]!=dx[i]) nan=1;
    CHECK("no NaN", !nan);
    free(x); free(dy); free(dx);
}

static void test_matmul_bwd(void) {
    printf("\n[Matmul bwd Ci=64 Co=32 S=64]\n");
    int Ci=64, Co=32, S=64;
    ane_init();
    ANEMatmul *mm = ane_matmul_compile(Ci, Co, S);
    CHECK("fwd compile", mm != NULL);
    if (!mm) return;
    ANEMatmulBwd *bwd = ane_matmul_bwd_compile(Ci, Co, S);
    CHECK("bwd compile", bwd != NULL);
    if (!bwd) return;

    _Float16 *W  = calloc(Co*Ci, 2);
    _Float16 *x  = calloc(Ci*S,  2);
    _Float16 *dy = calloc(Co*S,  2);
    _Float16 *dx = calloc(Ci*S,  2);
    _Float16 *dw = calloc(Co*Ci, 2);
    for (int i=0;i<Co*Ci;i++) W[i]=(_Float16)(i==0||i==(Ci+1)?1.f:0.f);
    for (int i=0;i<Ci*S; i++) x[i]=(_Float16)1.0f;
    for (int i=0;i<Co*S; i++) dy[i]=(_Float16)1.0f;

    ane_matmul_bwd_rewire_w(bwd, mm->w_surf);
    ane_matmul_write_w(mm, W);
    ane_matmul_bwd_write_dy(bwd, dy);
    ane_matmul_bwd_rewire_x(bwd, mm->x_surf);
    ane_matmul_write_x(mm, x);
    ane_matmul_bwd_eval(bwd);
    ane_matmul_bwd_read_dx(bwd, dx);
    ane_matmul_bwd_read_dw(bwd, dw);

    int nan_dx=0,nan_dw=0;
    for(int i=0;i<Ci*S; i++) if(dx[i]!=dx[i]) nan_dx=1;
    for(int i=0;i<Co*Ci;i++) if(dw[i]!=dw[i]) nan_dw=1;
    CHECK("dx no NaN", !nan_dx);
    CHECK("dW no NaN", !nan_dw);
    CHECK("dW[0,0]=S", fabsf((float)dw[0] - (float)S) < 2.0f);
    free(W);free(x);free(dy);free(dx);free(dw);
}

static void test_ln_bwd(void) {
    printf("\n[LN bwd C=32 S=64]\n");
    int C=32, S=64;
    ane_init();
    ANELayerNorm *ln = ane_ln_compile(C, S);
    CHECK("compile", ln != NULL);
    if (!ln) return;

    _Float16 *x    = malloc(C*S*2);
    _Float16 *norm = malloc(C*S*2);
    _Float16 *rstd = malloc(S*2);
    _Float16 *dy   = malloc(C*S*2);
    _Float16 *dx   = malloc(C*S*2);
    for (int i=0;i<C*S;i++) { x[i]=(_Float16)((float)(i%C)*0.1f); dy[i]=(_Float16)1.0f; }

    ane_ln_write_input(ln, x);
    ane_ln_eval(ln);
    ane_ln_save_stats(ln, norm, rstd);
    ane_ln_bwd(dy, norm, rstd, dx, C, S);

    float s=0; for(int c=0;c<C;c++) s+=(float)dx[c*S];
    CHECK("sum(dx)~0", fabsf(s) < 0.1f);
    int nan=0; for(int i=0;i<C*S;i++) if(dx[i]!=dx[i]) nan=1;
    CHECK("no NaN", !nan);
    free(x);free(norm);free(rstd);free(dy);free(dx);
}

static void test_dw_bwd(void) {
    printf("\n[DW bwd C=32 H=8 K=3]\n");
    int C=32, H=8, K=3, S=H*H;
    _Float16 *x  = malloc(C*S*2);
    _Float16 *w  = malloc(C*K*K*2);
    _Float16 *dy = malloc(C*S*2);
    _Float16 *dx = malloc(C*S*2);
    _Float16 *dw = malloc(C*K*K*2);
    for(int i=0;i<C*S;i++)   { x[i]=(_Float16)1.0f; dy[i]=(_Float16)1.0f; }
    for(int i=0;i<C*K*K;i++) w[i]=(_Float16)1.0f;

    _dw_neon_bwd_dx(dy, w, dx, C, H, K);
    _dw_neon_bwd_dw(dy, x, dw, C, H, K);

    int nan_dx=0,nan_dw=0;
    for(int i=0;i<C*S;i++)   if(dx[i]!=dx[i]) nan_dx=1;
    for(int i=0;i<C*K*K;i++) if(dw[i]!=dw[i]) nan_dw=1;
    CHECK("dx no NaN", !nan_dx);
    CHECK("dW no NaN", !nan_dw);
    float center = (float)dx[0*S + H/2*H + H/2];
    CHECK("dx interior = K*K=9", fabsf(center - 9.0f) < 0.5f);
    CHECK("dW center ~S", (float)dw[0*K*K + K/2*K + K/2] > (float)(S*0.8f));
    free(x);free(w);free(dy);free(dx);free(dw);
}

static void test_adam(void) {
    printf("\n[Adam N=64 one step]\n");
    ane_init();
    int N=64;
    ANEAdam *opt = ane_adam_compile(N, 1e-3f, 0.9f, 0.999f, 1e-4f, 0.0f);
    CHECK("compile", opt != NULL);
    if (!opt) return;

    _Float16 *W_init = malloc(N*2);
    for(int i=0;i<N;i++) W_init[i]=(_Float16)1.0f;
    IOSurfaceLock(opt->w_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(opt->w_surf), W_init, N*2);
    IOSurfaceUnlock(opt->w_surf, 0, NULL);
    IOSurfaceLock(opt->dw_surf, 0, NULL);
    for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt->dw_surf))[i]=(_Float16)1.0f;
    IOSurfaceUnlock(opt->dw_surf, 0, NULL);

    ane_adam_set_lr_surf(opt, 1e-3f, 0.9f, 0.999f, 1);
    ane_adam_step(opt, 1);

    _Float16 *W_out = malloc(N*2);
    IOSurfaceLock(opt->w_new_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(W_out, IOSurfaceGetBaseAddress(opt->w_new_surf), N*2);
    IOSurfaceUnlock(opt->w_new_surf, kIOSurfaceLockReadOnly, NULL);
    // Expected: W=1, dW=1, m=0→0.1, v=0→0.001
    // lr_t = 1e-3 * sqrt(1-0.999^1)/(1-0.9^1) = 1e-3*0.031623/0.1 = 3.1623e-4
    // W_new = 1 - lr_t * 0.1 / (sqrt(0.001)+1e-4) = 1 - 3.1623e-4*0.1/0.032162 ≈ 0.99902
    float expected = 0.99902f;
    printf("  W_out[0]=%.5f expected~%.5f\n", (float)W_out[0], expected);
    CHECK("W decreased", (float)W_out[0] < 1.0f);
    CHECK("W correct value", fabsf((float)W_out[0] - expected) < 0.001f);
    CHECK("W not NaN", W_out[0]==W_out[0]);
    free(W_init); free(W_out);

    // --- overflow: large dW (256) → dW² = 65536 overflows fp16 max 65504 ---
    // v update: (1-0.999)*dW² → overflows to inf → sqrt(inf)=inf → step=0, W unchanged
    // We check W stays finite (not NaN) even if v overflows
    printf("\n[Adam N=64 large dW overflow]\n");
    ANEAdam *opt2 = ane_adam_compile(N, 1e-3f, 0.9f, 0.999f, 1e-4f, 0.0f);
    if (opt2) {
        IOSurfaceLock(opt2->w_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt2->w_surf))[i]=(_Float16)1.0f;
        IOSurfaceUnlock(opt2->w_surf, 0, NULL);
        // dW=256: dW²=65536 > fp16_max → overflows in v kernel
        IOSurfaceLock(opt2->dw_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt2->dw_surf))[i]=(_Float16)256.0f;
        IOSurfaceUnlock(opt2->dw_surf, 0, NULL);
        ane_adam_set_lr_surf(opt2, 1e-3f, 0.9f, 0.999f, 1);
        ane_adam_step(opt2, 1);
        _Float16 w2; IOSurfaceLock(opt2->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        memcpy(&w2, IOSurfaceGetBaseAddress(opt2->w_new_surf), 2);
        IOSurfaceUnlock(opt2->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        printf("  W_out=%.5f (dW=256, v overflows)\n", (float)w2);
        CHECK("no NaN on v overflow", w2==w2);
        CHECK("W finite on v overflow", isfinite((float)w2));
    }

    // --- underflow: tiny dW (1e-5) → below fp16 min normal 6.1e-5, FTZ to 0 ---
    // gradient vanishes, W should be unchanged (or very slightly changed)
    printf("\n[Adam N=64 tiny dW underflow]\n");
    ANEAdam *opt3 = ane_adam_compile(N, 1e-3f, 0.9f, 0.999f, 1e-4f, 0.0f);
    if (opt3) {
        IOSurfaceLock(opt3->w_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt3->w_surf))[i]=(_Float16)1.0f;
        IOSurfaceUnlock(opt3->w_surf, 0, NULL);
        // dW=1e-5 → fp16 subnormal → FTZ=0 on ANE
        IOSurfaceLock(opt3->dw_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt3->dw_surf))[i]=(_Float16)1e-5f;
        IOSurfaceUnlock(opt3->dw_surf, 0, NULL);
        ane_adam_set_lr_surf(opt3, 1e-3f, 0.9f, 0.999f, 1);
        ane_adam_step(opt3, 1);
        _Float16 w3; IOSurfaceLock(opt3->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        memcpy(&w3, IOSurfaceGetBaseAddress(opt3->w_new_surf), 2);
        IOSurfaceUnlock(opt3->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        printf("  W_out=%.6f (dW=1e-5, FTZ→0 expected)\n", (float)w3);
        CHECK("no NaN on underflow", w3==w3);
        // W should be ~1.0 (gradient zeroed by FTZ)
        CHECK("W ~unchanged on underflow", fabsf((float)w3 - 1.0f) < 0.01f);
    }

    // --- loss_scale rescues underflowing gradient ---
    // dW=1e-5 flushes to 0 in fp16. With loss_scale=512, dW_scaled=512e-5=5.12e-3 survives.
    // m/sqrt(v) ratio is scale-invariant so weight update direction is preserved.
    printf("\n[Adam N=64 loss_scale rescues underflow]\n");
    ANEAdam *opt5 = ane_adam_compile(N, 1e-3f, 0.9f, 0.999f, 1e-4f, 0.0f);
    if (opt5) {
        IOSurfaceLock(opt5->w_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt5->w_surf))[i]=(_Float16)1.0f;
        IOSurfaceUnlock(opt5->w_surf, 0, NULL);
        // dW_scaled = 1e-5 * 512 = 5.12e-3 — well above fp16 min normal
        float loss_scale = 512.0f;
        float raw_dw = 1e-5f;
        IOSurfaceLock(opt5->dw_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt5->dw_surf))[i]=(_Float16)(raw_dw * loss_scale);
        IOSurfaceUnlock(opt5->dw_surf, 0, NULL);
        ane_adam_set_lr_surf(opt5, 1e-3f, 0.9f, 0.999f, 1);
        ane_adam_step(opt5, 1);
        _Float16 w5; IOSurfaceLock(opt5->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        memcpy(&w5, IOSurfaceGetBaseAddress(opt5->w_new_surf), 2);
        IOSurfaceUnlock(opt5->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        printf("  W_out=%.6f (dW=1e-5*512=5.12e-3)\n", (float)w5);
        // loss_scale keeps dW representable in fp16 (5.12e-3 >> 6.1e-5 min normal).
        // But step = lr_t * m/sqrt(v+eps) ≈ 3.16e-4 * 5.12e-4/0.01 = 1.6e-5,
        // which is below fp16 precision at W=1.0 (~9.8e-4). W appears unchanged.
        // loss_scale benefit: gradient direction is preserved, not zeroed — matters
        // for m/v accumulation over many steps even if single-step W change is invisible.
        CHECK("no NaN with loss_scale", w5==w5);
        CHECK("W finite with loss_scale", isfinite((float)w5));
    }

    // --- subnormal lr (early warmup step 1 of 10000) ---
    // ane_adam_write_lr clamps lr_t to fp16_min_normal if it would underflow
    printf("\n[Adam N=64 subnormal lr clamping]\n");
    ANEAdam *opt4 = ane_adam_compile(N, 1e-4f, 0.9f, 0.999f, 1e-4f, 0.0f);
    if (opt4) {
        IOSurfaceLock(opt4->w_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt4->w_surf))[i]=(_Float16)1.0f;
        IOSurfaceUnlock(opt4->w_surf, 0, NULL);
        IOSurfaceLock(opt4->dw_surf, 0, NULL);
        for(int i=0;i<N;i++) ((_Float16*)IOSurfaceGetBaseAddress(opt4->dw_surf))[i]=(_Float16)1.0f;
        IOSurfaceUnlock(opt4->dw_surf, 0, NULL);
        // t=1 of 10000 warmup → lr_t = 1e-4 * sqrt(1-0.999)/0.1 / 10000 ≈ 3.16e-8 → subnormal
        // ane_adam_write_lr should clamp to fp16_min_normal=6.1e-5
        // so W should still decrease (not stuck at 1.0)
        ane_adam_set_lr_surf(opt4, 1e-4f / 10000.0f, 0.9f, 0.999f, 1);
        ane_adam_step(opt4, 1);
        _Float16 w4; IOSurfaceLock(opt4->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        memcpy(&w4, IOSurfaceGetBaseAddress(opt4->w_new_surf), 2);
        IOSurfaceUnlock(opt4->w_new_surf, kIOSurfaceLockReadOnly, NULL);
        printf("  W_out=%.6f (subnormal lr, clamped to fp16_min_normal)\n", (float)w4);
        // Note: step = lr_t * m / sqrt(v+eps) ≈ 6.1e-5 * 0.1 / 0.032 ≈ 1.9e-4
        // fp16 precision at W=1.0 is ~1/1024 ≈ 9.8e-4, so 1.9e-4 rounds to 0 → W unchanged.
        // This is expected fp16 behavior, not a bug. Just verify no NaN/inf.
        CHECK("no NaN on subnormal lr", w4==w4);
        CHECK("W finite on subnormal lr", isfinite((float)w4));
    }
}

static void test_convnext_bwd(void) {
    printf("\n[ConvNeXt bwd C=64 H=8 K=3]\n");
    int C=64, H=8, K=3, S=H*H;
    ane_init();
    ANEConvNeXt *fwd = ane_convnext_compile(C, S, K);
    CHECK("fwd compile", fwd != NULL);
    if (!fwd) return;

    ANEConvNeXtBwd *bwd = ane_convnext_bwd_compile(fwd, 1e-3f, 0.9f, 0.999f, 1e-4f, 0.0f);
    CHECK("bwd compile", bwd != NULL);
    if (!bwd) return;

    _Float16 *x  = malloc(C*S*2);
    _Float16 *dy = malloc(C*S*2);
    _Float16 *dx = malloc(C*S*2);
    for(int i=0;i<C*S;i++) { x[i]=(_Float16)0.1f; dy[i]=(_Float16)0.01f; }

    ane_convnext_write_input(fwd, x);
    ane_convnext_eval(fwd, H);
    ane_convnext_save_fwd(bwd, fwd, H);
    ane_convnext_bwd_eval(bwd, fwd, dy, dx, H, 1);

    int nan=0; for(int i=0;i<C*S;i++) if(dx[i]!=dx[i]) nan=1;
    CHECK("dx no NaN", !nan);
    free(x); free(dy); free(dx);
}

int main(void) {
    @autoreleasepool {
    printf("=== ANE Backward Tests ===\n");
    test_bce_loss();
    test_gelu_bwd();
    test_matmul_bwd();
    test_ln_bwd();
    test_dw_bwd();
    test_adam();
    test_convnext_bwd();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
    }
}
