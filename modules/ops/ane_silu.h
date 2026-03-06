// ane_silu.h — ANE SiLU: y = x * sigmoid(x)
// 2 ops (sigmoid + mul) vs GELU's 9 — much faster on ANE.
// Same interface as ane_gelu.h for drop-in replacement.
#pragma once
#include "../../ane_runtime.h"

#define ANE_SILU_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef x_surf;
    IOSurfaceRef y_surf;
} ANESilu;

static ANESilu *ane_silu_compile(int C, int S) {
    size_t sn = (size_t)C * S * 2;  if (sn < 2048) sn = 2048;
    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_SILU_BI "\n{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> X) {\n"
        "    tensor<fp16,[1,%d,%d]> SIG=sigmoid(x=X)[name=tensor<string,[]>(\"SIG\")];\n"
        "    tensor<fp16,[1,%d,%d]> Y  =mul(x=X,y=SIG)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        C, S, C, S, C, S];
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 1, &sn, 1, &sn);
    if (!k) { fprintf(stderr, "ane_silu_compile FAILED (C=%d S=%d)\n", C, S); return NULL; }
    ANESilu *g = (ANESilu *)calloc(1, sizeof(ANESilu));
    g->C = C; g->S = S;
    g->k = k;
    g->x_surf = k->ioInputs[0];
    g->y_surf = k->ioOutputs[0];
    return g;
}

static void ane_silu_write_x(ANESilu *g, const _Float16 *x) {
    IOSurfaceLock(g->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(g->x_surf), x, g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->x_surf, 0, NULL);
}

static void ane_silu_eval(ANESilu *g) { ane_eval(g->k); }

static void ane_silu_read_y(ANESilu *g, _Float16 *y) {
    IOSurfaceLock(g->y_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(g->y_surf), g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->y_surf, kIOSurfaceLockReadOnly, NULL);
}
