// ane_silu_bwd.h — ANE SiLU backward: dx = dy * silu'(x)
//
// silu'(x) = sigma(x) * (1 + x*(1 - sigma(x)))
//          = sigma(x) + x*sigma(x)*(1 - sigma(x))
//
// MIL ops: sigmoid, mul, sub, mul, add, mul = 6 ops (vs GELU bwd's ~15)
//
// Inputs: dy [1,C,S], x [1,C,S] (saved from forward)
// Output: dx [1,C,S]
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_SILU_BWD_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef dy_surf;
    IOSurfaceRef x_surf;
    IOSurfaceRef dx_surf;
} ANESiluBwd;

static ANESiluBwd *ane_silu_bwd_compile(int C, int S) {
    size_t sn = (size_t)C * S * 2;  if (sn < 2048) sn = 2048;
    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_SILU_BWD_BI "\n{\n"
        "  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> dy,\n"
        "    tensor<fp16,[1,%d,%d]> x) {\n"
        "    tensor<fp16,[]> one=const()[name=tensor<string,[]>(\"one\"),val=tensor<fp16,[]>(1.0)];\n"
        // sig = sigmoid(x)
        "    tensor<fp16,[1,%d,%d]> sig =sigmoid(x=x)  [name=tensor<string,[]>(\"sig\")];\n"
        // osig = 1 - sig
        "    tensor<fp16,[1,%d,%d]> osig=sub(x=one,y=sig)[name=tensor<string,[]>(\"osig\")];\n"
        // xosig = x * osig
        "    tensor<fp16,[1,%d,%d]> xosig=mul(x=x,y=osig)[name=tensor<string,[]>(\"xosig\")];\n"
        // ip1 = 1 + xosig
        "    tensor<fp16,[1,%d,%d]> ip1 =add(x=one,y=xosig)[name=tensor<string,[]>(\"ip1\")];\n"
        // gpx = sig * ip1
        "    tensor<fp16,[1,%d,%d]> gpx =mul(x=sig,y=ip1)[name=tensor<string,[]>(\"gpx\")];\n"
        // dx = dy * gpx
        "    tensor<fp16,[1,%d,%d]> dx  =mul(x=dy, y=gpx)[name=tensor<string,[]>(\"dx\")];\n"
        "  } -> (dx);\n}\n",
        C,S, C,S,
        C,S, C,S, C,S, C,S, C,S, C,S];
    size_t ins[2] = {sn, sn};
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins, 1, &sn);
    if (!k) { fprintf(stderr, "ane_silu_bwd_compile FAILED (C=%d S=%d)\n", C, S); return NULL; }
    ANESiluBwd *g = (ANESiluBwd *)calloc(1, sizeof(ANESiluBwd));
    g->C = C; g->S = S; g->k = k;
    g->dy_surf = k->ioInputs[0];
    g->x_surf  = k->ioInputs[1];
    g->dx_surf = k->ioOutputs[0];
    return g;
}

static void ane_silu_bwd_write_dy(ANESiluBwd *g, const _Float16 *dy) {
    IOSurfaceLock(g->dy_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(g->dy_surf), dy, g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->dy_surf, 0, NULL);
}

static void ane_silu_bwd_write_x(ANESiluBwd *g, const _Float16 *x) {
    IOSurfaceLock(g->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(g->x_surf), x, g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->x_surf, 0, NULL);
}

static void ane_silu_bwd_eval(ANESiluBwd *g) { ane_eval(g->k); }

static void ane_silu_bwd_read_dx(ANESiluBwd *g, _Float16 *dx) {
    IOSurfaceLock(g->dx_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(dx, IOSurfaceGetBaseAddress(g->dx_surf), g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->dx_surf, kIOSurfaceLockReadOnly, NULL);
}
