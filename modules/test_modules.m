// test_modules.m — Tests for ANE ops and blocks
// Build: make -C .. test_modules   (or see bottom)
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -o test_modules test_modules.m
#import <Foundation/Foundation.h>
#include <math.h>
#include <time.h>
#include "ops/ane_ln.h"
#include "ops/ane_matmul.h"
#include "ops/ane_gelu.h"
#include "ops/ane_silu.h"
#include "ops/ane_add.h"
#include "ops/ane_dw.h"
#include "ops/ane_sigmoid.h"
#include "ops/ane_upsample2x.h"
#include "ops/ane_stem.h"
#include "ops/ane_down.h"
#include "ops/ane_fuse.h"
#include "blocks/ane_convnext.h"
#include "ops/ane_attn.h"
#include "ops/ane_mhattn.h"
// ane_unet_large.h included via ane_convnext.h transitively — no direct include needed

int g_fp16_io = 1;

static int g_pass = 0, g_fail = 0;

#define CHECK(label, cond) do { \
    if (cond) { printf("  PASS: %s\n", label); g_pass++; } \
    else       { printf("  FAIL: %s\n", label); g_fail++; } \
} while(0)

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6;
}

// ---- CPU refs ----

static void cpu_ln(const float *x, float *y, int C, int S, float eps) {
    for (int s = 0; s < S; s++) {
        float mn = 0, vr = 0;
        for (int c = 0; c < C; c++) mn += x[c*S+s]; mn /= C;
        for (int c = 0; c < C; c++) { float d=x[c*S+s]-mn; vr+=d*d; } vr /= C;
        float rs = 1.f/sqrtf(vr+eps);
        for (int c = 0; c < C; c++) y[c*S+s] = (x[c*S+s]-mn)*rs;
    }
}

static void cpu_matmul(const float *W, const float *x, float *y, int Ci, int Co, int S) {
    // W[Co,Ci] @ x[Ci,S] → y[Co,S]
    for (int co=0;co<Co;co++) for (int s=0;s<S;s++) {
        float acc=0; for (int ci=0;ci<Ci;ci++) acc+=W[co*Ci+ci]*x[ci*S+s];
        y[co*S+s]=acc;
    }
}

static float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float cpu_gelu(float x) {
    float c0=0.7978845608f, c1=0.044715f;
    float u = c0*(x+c1*x*x*x);
    return 0.5f*x*(1.f+tanhf(u));
}

static float max_err_f16(const float *ref, const _Float16 *ane, int N) {
    float me=0;
    for (int i=0;i<N;i++) { float e=fabsf(ref[i]-(float)ane[i]); if(e>me)me=e; }
    return me;
}

// ---- Tests ----

static void test_ln(void) {
    printf("\n[LayerNorm C=96 S=1024]\n");
    int C=96, S=1024;
    ane_init();
    ANELayerNorm *ln = ane_ln_compile(C, S);
    CHECK("compile", ln != NULL);
    if (!ln) return;

    srand(1);
    float *xf=malloc(C*S*4), *yref=malloc(C*S*4);
    _Float16 *xh=malloc(C*S*2), *yh=malloc(C*S*2);
    for (int i=0;i<C*S;i++) { xf[i]=(float)rand()/RAND_MAX*2-1; xh[i]=(_Float16)xf[i]; }

    cpu_ln(xf, yref, C, S, 1e-5f);
    ane_ln_write_input(ln, xh);
    ane_ln_eval(ln);
    ane_ln_read_output(ln, yh);

    CHECK("max_err < 0.01", max_err_f16(yref, yh, C*S) < 0.01f);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_ln_eval(ln);
    printf("  perf: %.3f ms/call (200 reps)\n", (now_ms()-t0)/200);

    free(xf);free(yref);free(xh);free(yh);
}

static void test_matmul(void) {
    printf("\n[Matmul Ci=96 Co=384 S=1024]\n");
    int Ci=96, Co=384, S=1024;
    ane_init();
    ANEMatmul *mm = ane_matmul_compile(Ci, Co, S);
    CHECK("compile", mm != NULL);
    if (!mm) return;

    srand(2);
    float *Wf=malloc(Co*Ci*4), *xf=malloc(Ci*S*4), *yref=malloc(Co*S*4);
    _Float16 *Wh=malloc(Co*Ci*2), *xh=malloc(Ci*S*2), *yh=malloc(Co*S*2);
    for (int i=0;i<Co*Ci;i++) { Wf[i]=(float)rand()/RAND_MAX*0.1f; Wh[i]=(_Float16)Wf[i]; }
    for (int i=0;i<Ci*S;i++) { xf[i]=(float)rand()/RAND_MAX*2-1;   xh[i]=(_Float16)xf[i]; }

    cpu_matmul(Wf, xf, yref, Ci, Co, S);
    ane_matmul_write_w(mm, Wh);
    ane_matmul_write_x(mm, xh);
    ane_matmul_eval(mm);
    ane_matmul_read_y(mm, yh);

    CHECK("max_err < 0.05", max_err_f16(yref, yh, Co*S) < 0.05f);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_matmul_eval(mm);
    printf("  perf: %.3f ms/call (200 reps)\n", (now_ms()-t0)/200);

    free(Wf);free(xf);free(yref);free(Wh);free(xh);free(yh);
}

static void test_gelu(void) {
    printf("\n[GELU C=384 S=1024]\n");
    int C=384, S=1024;
    ane_init();
    ANEGelu *g = ane_gelu_compile(C, S);
    CHECK("compile", g != NULL);
    if (!g) return;

    srand(3);
    int N=C*S;
    float *xf=malloc(N*4), *yref=malloc(N*4);
    _Float16 *xh=malloc(N*2), *yh=malloc(N*2);
    for (int i=0;i<N;i++) { xf[i]=(float)rand()/RAND_MAX*4-2; xh[i]=(_Float16)xf[i]; }
    for (int i=0;i<N;i++) yref[i]=cpu_gelu(xf[i]);

    ane_gelu_write_x(g, xh);
    ane_gelu_eval(g);
    ane_gelu_read_y(g, yh);

    CHECK("max_err < 0.01", max_err_f16(yref, yh, N) < 0.01f);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_gelu_eval(g);
    printf("  perf: %.3f ms/call (200 reps)\n", (now_ms()-t0)/200);

    free(xf);free(yref);free(xh);free(yh);
}

static void test_gelu_cpu(void) {
    printf("\n[GELU ANE C=384 S=4096 (large, should use ANE not CPU)]\n");
    int C=384, S=4096;
    ane_init();
    ANEGelu *g = ane_gelu_compile(C, S);
    CHECK("compile", g != NULL);
    if (!g) return;

    srand(7);
    int N=C*S;
    float *xf=malloc(N*4), *yref=malloc(N*4);
    _Float16 *xh=malloc(N*2), *yh=malloc(N*2);
    for (int i=0;i<N;i++) { xf[i]=(float)rand()/RAND_MAX*4-2; xh[i]=(_Float16)xf[i]; }
    for (int i=0;i<N;i++) yref[i]=cpu_gelu(xf[i]);

    ane_gelu_write_x(g, xh);
    ane_gelu_eval(g);  // takes CPU path
    ane_gelu_read_y(g, yh);

    CHECK("max_err < 0.01", max_err_f16(yref, yh, N) < 0.01f);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_gelu_eval(g);
    printf("  perf: %.3f ms/call (200 reps, ANE large)\n", (now_ms()-t0)/200);

    free(xf);free(yref);free(xh);free(yh);
}

static void test_silu(void) {
    printf("\n[SiLU ANE C=384 S=1024]\n");
    int C=384, S=1024;
    ane_init();
    ANESilu *g = ane_silu_compile(C, S);
    CHECK("compile", g != NULL);
    if (!g) return;

    srand(5);
    int N=C*S;
    float *xf=malloc(N*4), *yref=malloc(N*4);
    _Float16 *xh=malloc(N*2), *yh=malloc(N*2);
    for (int i=0;i<N;i++) { xf[i]=(float)rand()/RAND_MAX*4-2; xh[i]=(_Float16)xf[i]; }
    // ref from fp16 inputs (same as ANE sees)
    for (int i=0;i<N;i++) yref[i]=cpu_silu((float)xh[i]);

    ane_silu_write_x(g, xh);
    ane_silu_eval(g);
    ane_silu_read_y(g, yh);

    float silu_err = max_err_f16(yref, yh, N);
    printf("  max_err=%.5f\n", silu_err);
    CHECK("max_err < 0.02", silu_err < 0.02f);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_silu_eval(g);
    printf("  perf: %.3f ms/call (200 reps, ANE)\n", (now_ms()-t0)/200);

    // Also test larger size to compare with GELU CPU path
    printf("\n[SiLU ANE C=384 S=4096]\n");
    ANESilu *g2 = ane_silu_compile(384, 4096);
    CHECK("compile large", g2 != NULL);
    if (g2) {
        t0=now_ms();
        for (int i=0;i<200;i++) ane_silu_eval(g2);
        printf("  perf: %.3f ms/call (200 reps, ANE large)\n", (now_ms()-t0)/200);
    }

    free(xf);free(yref);free(xh);free(yh);
}

static void test_add(void) {
    printf("\n[Add C=96 S=1024]\n");
    int C=96, S=1024;
    ane_init();
    ANEAdd *add = ane_add_compile(C, S);
    CHECK("compile", add != NULL);
    if (!add) return;

    srand(4);
    int N=C*S;
    float *af=malloc(N*4), *bf=malloc(N*4), *cref=malloc(N*4);
    _Float16 *ah=malloc(N*2), *bh=malloc(N*2), *ch=malloc(N*2);
    for (int i=0;i<N;i++) { af[i]=(float)rand()/RAND_MAX*2-1; ah[i]=(_Float16)af[i]; }
    for (int i=0;i<N;i++) { bf[i]=(float)rand()/RAND_MAX*2-1; bh[i]=(_Float16)bf[i]; }
    for (int i=0;i<N;i++) cref[i]=af[i]+bf[i];

    IOSurfaceLock(add->a_surf,0,NULL);
    memcpy(IOSurfaceGetBaseAddress(add->a_surf),ah,N*2);
    IOSurfaceUnlock(add->a_surf,0,NULL);
    IOSurfaceLock(add->b_surf,0,NULL);
    memcpy(IOSurfaceGetBaseAddress(add->b_surf),bh,N*2);
    IOSurfaceUnlock(add->b_surf,0,NULL);

    ane_add_eval(add);
    ane_add_read_c(add, ch);

    CHECK("max_err < 0.01", max_err_f16(cref, ch, N) < 0.01f);

    free(af);free(bf);free(cref);free(ah);free(bh);free(ch);
}

static void test_dw(void) {
    printf("\n[Depthwise C=96 H=32 K=7 (S=1024)]\n");
    int C=96, H=32, S=H*H, K=7;
    ane_init();
    ANEDepthwise *dw = ane_dw_compile(C, S, H, K);
    CHECK("compile", dw != NULL);
    if (!dw) return;

    srand(6);
    _Float16 *x  = malloc(S*C*2);
    _Float16 *w  = malloc(C*K*K*2);
    _Float16 *yref = malloc(S*C*2);
    _Float16 *yane = malloc(S*C*2);
    for (int i=0;i<S*C;i++)  x[i] =(_Float16)((float)rand()/RAND_MAX*2-1);
    for (int i=0;i<C*K*K;i++) w[i]=(_Float16)((float)rand()/RAND_MAX*2-1)*0.1f;

    // CPU ref in [S,C] layout
    float *yf = calloc(S*C, 4);
    for (int ky=0;ky<K;ky++) for (int kx=0;kx<K;kx++) {
        int k=ky*K+kx, dy=ky-K/2, dx=kx-K/2;
        for (int h=0;h<H;h++) for (int ww=0;ww<H;ww++) {
            int ih=h+dy, iw=ww+dx;
            if (ih>=0&&ih<H&&iw>=0&&iw<H)
                for (int c=0;c<C;c++)
                    yf[(h*H+ww)*C+c]+=(float)w[c*K*K+k]*(float)x[(ih*H+iw)*C+c];
        }
    }

    ane_dw_write_w(dw, w);
    ane_dw_write_input(dw, x);
    ane_dw_eval(dw);
    ane_dw_read_output(dw, yane);

    float me=0;
    for (int i=0;i<S*C;i++){float e=fabsf((float)yane[i]-yf[i]);if(e>me)me=e;}
    printf("  max_err=%.4f\n", me);
    CHECK("max_err < 0.05", me < 0.05f);  // fp16 accumulation over 49 terms
    free(yf);

    double t0=now_ms();
    for (int i=0;i<50;i++) ane_dw_eval(dw);
    printf("  perf: %.3f ms/call (50 reps, eval only)\n", (now_ms()-t0)/50);

    free(x); free(w); free(yref); free(yane);
}

static void test_convnext(void) {
    printf("\n[ConvNeXt C=96 H=32 K=7]\n");
    int C=96, H=32, S=H*H, K=7;
    ane_init();
    ANEConvNeXt *blk = ane_convnext_compile(C, S, K);
    CHECK("compile", blk != NULL);
    if (!blk) return;

    srand(5);
    // tiny weights so values don't explode
    _Float16 *dw  = malloc(C*K*K*2);
    _Float16 *pw1 = malloc(C*4*C*2);
    _Float16 *pw2 = malloc(C*C*4*2);
    _Float16 *xh  = malloc(C*S*2);
    _Float16 *yh  = malloc(C*S*2);
    float scale = 0.01f;
    for (int i=0;i<C*K*K;i++) dw[i] =(_Float16)((float)rand()/RAND_MAX*2-1)*scale;
    for (int i=0;i<C*4*C;i++) pw1[i]=(_Float16)((float)rand()/RAND_MAX*2-1)*scale;
    for (int i=0;i<C*C*4;i++) pw2[i]=(_Float16)((float)rand()/RAND_MAX*2-1)*scale;
    for (int i=0;i<C*S;  i++) xh[i] =(_Float16)((float)rand()/RAND_MAX*2-1);

    ane_convnext_set_weights(blk, dw, pw1, pw2);
    ane_convnext_write_input(blk, xh);
    ane_convnext_eval(blk, H);
    ane_convnext_read_output(blk, yh);

    // sanity: output not all zero, not NaN
    float sum=0; int has_nan=0;
    for (int i=0;i<C*S;i++) { float v=(float)yh[i]; sum+=fabsf(v); if(v!=v)has_nan=1; }
    CHECK("output not zero", sum > 0.f);
    CHECK("no NaN", !has_nan);
    // with tiny pw2 weights, output ≈ input (residual dominates)
    float res_err=0;
    for (int i=0;i<C*S;i++) res_err+=fabsf((float)yh[i]-(float)xh[i]);
    res_err/=C*S;
    printf("  mean |y-x|=%.4f (should be small with scale=0.01)\n", res_err);
    CHECK("residual dominates (mean diff < 0.1)", res_err < 0.1f);

    double t0=now_ms();
    for (int i=0;i<50;i++) ane_convnext_eval(blk, H);
    printf("  perf: %.3f ms/call (50 reps)\n", (now_ms()-t0)/50);

    free(dw);free(pw1);free(pw2);free(xh);free(yh);
}

static void test_sigmoid(void) {
    printf("\n[Sigmoid C=96 S=1024]\n");
    int C=96, S=1024;
    ane_init();
    ANESigmoid *sg = ane_sigmoid_compile(C, S);
    CHECK("compile", sg != NULL);
    if (!sg) return;

    srand(7);
    int N = C*S;
    float *xf = malloc(N*4), *yref = malloc(N*4);
    _Float16 *xh = malloc(N*2), *yh = malloc(N*2);
    for (int i=0;i<N;i++) { xf[i]=(float)rand()/RAND_MAX*6-3; xh[i]=(_Float16)xf[i]; }
    for (int i=0;i<N;i++) yref[i] = 1.f/(1.f+expf(-xf[i]));

    ane_sigmoid_write_x(sg, xh);
    ane_sigmoid_eval(sg);
    ane_sigmoid_read_y(sg, yh);

    CHECK("max_err < 0.01", max_err_f16(yref, yh, N) < 0.01f);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_sigmoid_eval(sg);
    printf("  perf: %.3f ms/call (200 reps)\n", (now_ms()-t0)/200);

    free(xf);free(yref);free(xh);free(yh);
}

static void test_upsample2x(void) {
    printf("\n[Upsample2x C=96 H=32 -> H=64 (CPU)]\n");
    int C=96, H=32, H2=H*2, S=H*H, S4=4*S;
    srand(8);
    _Float16 *xh = malloc(C*S*2), *yh = malloc(C*S4*2);
    for (int i=0;i<C*S;i++) xh[i]=(_Float16)((float)rand()/RAND_MAX*2-1);

    ane_upsample2x(xh, yh, C, H);

    float me=0;
    for (int c=0;c<C;c++) for (int iy=0;iy<H;iy++) for (int ix=0;ix<H;ix++) {
        float v = (float)xh[c*S + iy*H + ix];
        float e0 = fabsf((float)yh[c*S4 + (2*iy)*H2   + (2*ix)]   - v);
        float e1 = fabsf((float)yh[c*S4 + (2*iy)*H2   + (2*ix+1)] - v);
        float e2 = fabsf((float)yh[c*S4 + (2*iy+1)*H2 + (2*ix)]   - v);
        float e3 = fabsf((float)yh[c*S4 + (2*iy+1)*H2 + (2*ix+1)] - v);
        float em = e0>e1?e0:e1; em=em>e2?em:e2; em=em>e3?em:e3;
        if (em>me) me=em;
    }
    CHECK("max_err < 0.001", me < 0.001f);

    double t0=now_ms();
    for (int i=0;i<500;i++) ane_upsample2x(xh, yh, C, H);
    printf("  perf: %.3f ms/call (500 reps)\n", (now_ms()-t0)/500);

    free(xh); free(yh);
}

static void test_stem(void) {
    printf("\n[Stem Cin=3 Cout=96 H=64 K=4]\n");
    int Cin=3, Cout=96, H=64, K=4;  // H=64 -> Ho=16, S_out=256 > Cout=96 ✓
    ane_init();
    ANEStem *stem = ane_stem_compile(Cin, Cout, H, K);
    CHECK("compile", stem != NULL);
    if (!stem) return;

    int Ho=H/K, S_out=Ho*Ho, Cin_pad=stem->Cin_pad;
    srand(9);
    _Float16 *wh = calloc(Cout*Cin_pad, 2);
    _Float16 *xh = malloc(Cin*H*H*2);
    _Float16 *yh = malloc(Cout*S_out*2);
    // Small weights so values don't blow up
    for (int i=0;i<Cout*Cin_pad;i++) wh[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
    for (int i=0;i<Cin*H*H;i++) xh[i]=(_Float16)((float)rand()/RAND_MAX*2-1);

    ane_stem_write_w(stem, wh);
    ane_stem_eval(stem, xh);
    ane_stem_read_y(stem, yh);

    // Sanity: output not all zero, no NaN
    float sum=0; int has_nan=0;
    for (int i=0;i<Cout*S_out;i++){float v=(float)yh[i];sum+=fabsf(v);if(v!=v)has_nan=1;}
    CHECK("output not zero", sum > 0.f);
    CHECK("no NaN", !has_nan);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_stem_eval(stem, xh);
    printf("  perf: %.3f ms/call (200 reps, im2col+matmul)\n", (now_ms()-t0)/200);

    free(wh); free(xh); free(yh);
}

static void test_down(void) {
    printf("\n[Down Cin=96 Cout=192 H=64]\n");
    int Cin=96, Cout=192, H=64;
    ane_init();
    ANEDown *dn = ane_down_compile(Cin, Cout, H);
    CHECK("compile", dn != NULL);
    if (!dn) return;

    int S_out=dn->S_out, Cin_pad=dn->Cin_pad;
    srand(10);
    _Float16 *wh = calloc(Cout*Cin_pad, 2);
    _Float16 *xh = malloc(Cin*H*H*2);
    _Float16 *yh = malloc(Cout*S_out*2);
    for (int i=0;i<Cout*Cin_pad;i++) wh[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
    for (int i=0;i<Cin*H*H;i++) xh[i]=(_Float16)((float)rand()/RAND_MAX*2-1);

    ane_down_write_w(dn, wh);
    ane_down_eval(dn, xh);
    ane_down_read_y(dn, yh);

    float sum=0; int has_nan=0;
    for (int i=0;i<Cout*S_out;i++){float v=(float)yh[i];sum+=fabsf(v);if(v!=v)has_nan=1;}
    CHECK("output not zero", sum > 0.f);
    CHECK("no NaN", !has_nan);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_down_eval(dn, xh);
    printf("  perf: %.3f ms/call (200 reps, im2col+matmul)\n", (now_ms()-t0)/200);

    free(wh); free(xh); free(yh);
}

static void test_fuse(void) {
    printf("\n[Fuse Cin=288 Cout=96 S=4096]\n");
    int Cskip=96, Cup=192, Cin=288, Cout=96, S=4096;
    ane_init();
    ANEFuse *fuse = ane_fuse_compile(Cin, Cout, S);
    CHECK("compile", fuse != NULL);
    if (!fuse) return;

    srand(11);
    _Float16 *wh   = calloc(Cout*Cin, 2);
    _Float16 *skip = malloc(Cskip*S*2);
    _Float16 *up   = malloc(Cup*S*2);
    _Float16 *yh   = malloc(Cout*S*2);
    for (int i=0;i<Cout*Cin;i++) wh[i]=(_Float16)((float)rand()/RAND_MAX*0.05f-0.025f);
    for (int i=0;i<Cskip*S;i++) skip[i]=(_Float16)((float)rand()/RAND_MAX*2-1);
    for (int i=0;i<Cup*S;i++)   up[i]  =(_Float16)((float)rand()/RAND_MAX*2-1);

    ane_fuse_write_w(fuse, wh);
    ane_fuse_eval(fuse, skip, up, Cskip, Cup);
    ane_fuse_read_y(fuse, yh);

    float sum=0; int has_nan=0;
    for (int i=0;i<Cout*S;i++){float v=(float)yh[i];sum+=fabsf(v);if(v!=v)has_nan=1;}
    CHECK("output not zero", sum > 0.f);
    CHECK("no NaN", !has_nan);

    double t0=now_ms();
    for (int i=0;i<200;i++) ane_fuse_eval(fuse, skip, up, Cskip, Cup);
    printf("  perf: %.3f ms/call (200 reps)\n", (now_ms()-t0)/200);

    free(wh); free(skip); free(up); free(yh);
}

static void test_attn(void) {
    printf("\n[Attn C=64 S=256]\n");
    int C=64, S=256;
    ane_init();
    ANEAttn *attn = ane_attn_compile(C, S);
    CHECK("compile", attn != NULL);
    if (!attn) return;

    srand(42);
    int CC=C*C, CS=C*S;
    _Float16 *Wq=calloc(CC,2), *Wk=calloc(CC,2), *Wv=calloc(CC,2), *Wo=calloc(CC,2);
    _Float16 *xh=malloc(CS*2), *yh=malloc(CS*2);
    // Small weights to avoid fp16 overflow
    for (int i=0;i<CC;i++) {
        Wq[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
        Wk[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
        Wv[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
        Wo[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
    }
    for (int i=0;i<CS;i++) xh[i]=(_Float16)((float)rand()/RAND_MAX*2-1);

    ane_attn_set_weights(attn, Wq, Wk, Wv, Wo);
    ane_attn_write_x(attn, xh);
    ane_attn_eval(attn);
    ane_attn_read_y(attn, yh);

    float sum=0; int has_nan=0;
    for (int i=0;i<CS;i++){float v=(float)yh[i];sum+=fabsf(v);if(v!=v)has_nan=1;}
    CHECK("output not zero", sum > 0.f);
    CHECK("no NaN", !has_nan);

    double t0=now_ms();
    for (int i=0;i<50;i++) ane_attn_eval(attn);
    printf("  perf: %.3f ms/call (50 reps)\n", (now_ms()-t0)/50);

    free(Wq); free(Wk); free(Wv); free(Wo); free(xh); free(yh);
}

static void test_mhattn(void) {
    printf("\n[MHAttn C=96 nH=3 d=32 S=256]\n");
    int C=96, nH=3, S=256;
    ane_init();
    ANEMHAttn *mha = ane_mhattn_compile(C, nH, S);
    CHECK("compile", mha != NULL);
    if (!mha) return;

    srand(7);
    int CC=C*C, CS=C*S;
    _Float16 *Wq=calloc(CC,2),*Wk=calloc(CC,2),*Wv=calloc(CC,2),*Wo=calloc(CC,2);
    _Float16 *xh=malloc(CS*2), *yh=malloc(CS*2);
    for (int i=0;i<CC;i++){
        Wq[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
        Wk[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
        Wv[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
        Wo[i]=(_Float16)((float)rand()/RAND_MAX*0.1f-0.05f);
    }
    for (int i=0;i<CS;i++) xh[i]=(_Float16)((float)rand()/RAND_MAX*2-1);

    ane_mhattn_set_weights(mha, Wq, Wk, Wv, Wo);
    ane_mhattn_write_x(mha, xh);
    ane_mhattn_eval(mha);
    ane_mhattn_read_y(mha, yh);

    float sum=0; int has_nan=0;
    for (int i=0;i<CS;i++){float v=(float)yh[i];sum+=fabsf(v);if(v!=v)has_nan=1;}
    CHECK("output not zero", sum > 0.f);
    CHECK("no NaN", !has_nan);

    double t0=now_ms();
    for (int i=0;i<50;i++) ane_mhattn_eval(mha);
    printf("  perf: %.3f ms/call (50 reps)\n", (now_ms()-t0)/50);

    free(Wq);free(Wk);free(Wv);free(Wo);free(xh);free(yh);
}

// test_unet removed — use test_unet_large.m for ane_unet_large
// Backward tests → see test_bwd.m

int main(void) {
    @autoreleasepool {
    printf("=== ANE Module Tests ===\n");
    test_ln();
    test_matmul();
    test_gelu();
    test_gelu_cpu();
    test_silu();
    test_add();
    test_dw();
    test_sigmoid();
    test_upsample2x();
    test_stem();
    test_down();
    test_fuse();
    test_convnext();
    test_attn();
    test_mhattn();
    // test_unet() removed — use test_unet_large.m for ane_unet_large
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
    }
}
