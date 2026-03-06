// ane_gelu.h — ANE GELU (tanh approx): y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// Layout: [1, C, S] — same as all other ops.
//
// Usage:
//   ANEGelu *g = ane_gelu_compile(C, S);
//   ane_gelu_write_x(g, x_fp16);    // [C*S] fp16
//   ane_gelu_eval(g);
//   ane_gelu_read_y(g, y_fp16);     // [C*S] fp16
//   // or chain: g->x_surf / g->y_surf
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_GELU_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef x_surf;
    IOSurfaceRef y_surf;
} ANEGelu;

static ANEGelu *ane_gelu_compile(int C, int S) {
    size_t sn = (size_t)C * S * 2;  if (sn < 2048) sn = 2048;
    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_GELU_BI "\n{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> X) {\n"
        "    tensor<fp16,[]> half=const()[name=tensor<string,[]>(\"half\"),val=tensor<fp16,[]>(0.5)];\n"
        "    tensor<fp16,[]> one =const()[name=tensor<string,[]>(\"one\"), val=tensor<fp16,[]>(1.0)];\n"
        "    tensor<fp16,[]> c0  =const()[name=tensor<string,[]>(\"c0\"),  val=tensor<fp16,[]>(0.7978845608)];\n"
        "    tensor<fp16,[]> c1  =const()[name=tensor<string,[]>(\"c1\"),  val=tensor<fp16,[]>(0.044715)];\n"
        "    tensor<fp16,[1,%d,%d]> X3  =mul(x=X,  y=X) [name=tensor<string,[]>(\"X2\")];\n"
        "    tensor<fp16,[1,%d,%d]> X3b =mul(x=X3, y=X) [name=tensor<string,[]>(\"X3\")];\n"
        "    tensor<fp16,[1,%d,%d]> CX3 =mul(x=c1, y=X3b)[name=tensor<string,[]>(\"CX3\")];\n"
        "    tensor<fp16,[1,%d,%d]> INN =add(x=X,  y=CX3)[name=tensor<string,[]>(\"INN\")];\n"
        "    tensor<fp16,[1,%d,%d]> ARG =mul(x=c0, y=INN)[name=tensor<string,[]>(\"ARG\")];\n"
        "    tensor<fp16,[1,%d,%d]> TH  =tanh(x=ARG)    [name=tensor<string,[]>(\"TH\")];\n"
        "    tensor<fp16,[1,%d,%d]> TP1 =add(x=one,y=TH) [name=tensor<string,[]>(\"TP1\")];\n"
        "    tensor<fp16,[1,%d,%d]> XH  =mul(x=X,  y=TP1)[name=tensor<string,[]>(\"XH\")];\n"
        "    tensor<fp16,[1,%d,%d]> Y   =mul(x=half,y=XH)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        C, S,
        C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S];
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 1, &sn, 1, &sn);
    if (!k) { fprintf(stderr, "ane_gelu_compile FAILED (C=%d S=%d)\n", C, S); return NULL; }
    ANEGelu *g = (ANEGelu *)calloc(1, sizeof(ANEGelu));
    g->C = C; g->S = S;
    g->k = k;
    g->x_surf = k->ioInputs[0];
    g->y_surf = k->ioOutputs[0];
    return g;
}

static void ane_gelu_write_x(ANEGelu *g, const _Float16 *x) {
    IOSurfaceLock(g->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(g->x_surf), x, g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->x_surf, 0, NULL);
}

// CPU NEON GELU: y = 0.5*x*(1 + tanh(0.7978845608*(x + 0.044715*x^3)))
// Faster than ANE dispatch for large tensors (avoids ~1.5ms round-trip overhead).
#define _GELU_CPU_THRESHOLD (50*1024)
static void _ane_gelu_eval_cpu(ANEGelu *g) {
    int N = g->C * g->S;
    IOSurfaceLock(g->x_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(g->y_surf, 0, NULL);
    const _Float16 *x = (const _Float16 *)IOSurfaceGetBaseAddress(g->x_surf);
    _Float16       *y = (_Float16 *)IOSurfaceGetBaseAddress(g->y_surf);
    const float c0 = 0.7978845608f, c1 = 0.044715f;
    for (int i = 0; i < N; i++) {
        float xi = (float)x[i];
        float arg = c0 * (xi + c1 * xi * xi * xi);
        float t = tanhf(arg);
        y[i] = (_Float16)(0.5f * xi * (1.0f + t));
    }
    IOSurfaceUnlock(g->y_surf, 0, NULL);
    IOSurfaceUnlock(g->x_surf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_gelu_eval(ANEGelu *g) { ane_eval(g->k); }

static void ane_gelu_read_y(ANEGelu *g, _Float16 *y) {
    IOSurfaceLock(g->y_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(g->y_surf), g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->y_surf, kIOSurfaceLockReadOnly, NULL);
}
