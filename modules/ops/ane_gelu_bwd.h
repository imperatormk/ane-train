// ane_gelu_bwd.h — ANE GELU backward: dx = dy * gelu'(x)
//
// gelu'(x) = 0.5*(1+tanh(c0*(x+c1*x^3))) + 0.5*x*(1-tanh^2(...))*c0*(1+3*c1*x^2)
//          = 0.5*(1+T) + 0.5*x*(1-T^2)*c0*(1+3*c1*x^2)
//   where T = tanh(c0*(x+c1*x^3)), c0=sqrt(2/pi), c1=0.044715
//
// Inputs: x [1,C,S] (saved from forward), dy [1,C,S]
// Output: dx [1,C,S]
//
// Slot rule: both inputs same size [C*S], ANE slot0=dy slot1=x (larger arg second).
// Since they're equal size, order doesn't matter — we put dy first.
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_GELU_BWD_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef dy_surf;  // slot 0
    IOSurfaceRef x_surf;   // slot 1 (saved fwd input)
    IOSurfaceRef dx_surf;  // output
} ANEGeluBwd;

static ANEGeluBwd *ane_gelu_bwd_compile(int C, int S) {
    size_t sn = (size_t)C * S * 2;  if (sn < 2048) sn = 2048;
    // c0 = sqrt(2/pi) ≈ 0.7978845608, c1 = 0.044715
    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_GELU_BWD_BI "\n{\n"
        "  func main<ios16>(\n"
        "    tensor<fp16,[1,%d,%d]> dy,\n"
        "    tensor<fp16,[1,%d,%d]> x) {\n"
        "    tensor<fp16,[]> half =const()[name=tensor<string,[]>(\"half\"), val=tensor<fp16,[]>(0.5)];\n"
        "    tensor<fp16,[]> one  =const()[name=tensor<string,[]>(\"one\"),  val=tensor<fp16,[]>(1.0)];\n"
        "    tensor<fp16,[]> c0   =const()[name=tensor<string,[]>(\"c0\"),   val=tensor<fp16,[]>(0.7978845608)];\n"
        "    tensor<fp16,[]> c1   =const()[name=tensor<string,[]>(\"c1\"),   val=tensor<fp16,[]>(0.044715)];\n"
        "    tensor<fp16,[]> three=const()[name=tensor<string,[]>(\"three\"),val=tensor<fp16,[]>(3.0)];\n"
        // T = tanh(c0*(x + c1*x^3))
        "    tensor<fp16,[1,%d,%d]> x2  =mul(x=x,  y=x)[name=tensor<string,[]>(\"x2\")];\n"
        "    tensor<fp16,[1,%d,%d]> x3  =mul(x=x2, y=x)[name=tensor<string,[]>(\"x3\")];\n"
        "    tensor<fp16,[1,%d,%d]> cx3 =mul(x=c1, y=x3)[name=tensor<string,[]>(\"cx3\")];\n"
        "    tensor<fp16,[1,%d,%d]> inn =add(x=x,  y=cx3)[name=tensor<string,[]>(\"inn\")];\n"
        "    tensor<fp16,[1,%d,%d]> arg =mul(x=c0, y=inn)[name=tensor<string,[]>(\"arg\")];\n"
        "    tensor<fp16,[1,%d,%d]> T   =tanh(x=arg)[name=tensor<string,[]>(\"T\")];\n"
        // gelu'(x) = 0.5*(1+T) + 0.5*x*(1-T^2)*c0*(1+3*c1*x^2)
        "    tensor<fp16,[1,%d,%d]> Tp1 =add(x=one,y=T)[name=tensor<string,[]>(\"Tp1\")];\n"
        "    tensor<fp16,[1,%d,%d]> hTp1=mul(x=half,y=Tp1)[name=tensor<string,[]>(\"hTp1\")];\n"
        "    tensor<fp16,[1,%d,%d]> T2  =mul(x=T,  y=T)[name=tensor<string,[]>(\"T2\")];\n"
        "    tensor<fp16,[1,%d,%d]> oT2 =sub(x=one,y=T2)[name=tensor<string,[]>(\"oT2\")];\n"
        "    tensor<fp16,[1,%d,%d]> c1x2=mul(x=c1, y=x2)[name=tensor<string,[]>(\"c1x2\")];\n"
        "    tensor<fp16,[1,%d,%d]> tc1x2=mul(x=three,y=c1x2)[name=tensor<string,[]>(\"tc1x2\")];\n"
        "    tensor<fp16,[1,%d,%d]> ip1 =add(x=one,y=tc1x2)[name=tensor<string,[]>(\"ip1\")];\n"
        "    tensor<fp16,[1,%d,%d]> xoT2=mul(x=x,  y=oT2)[name=tensor<string,[]>(\"xoT2\")];\n"
        "    tensor<fp16,[1,%d,%d]> sc  =mul(x=c0, y=ip1)[name=tensor<string,[]>(\"sc\")];\n"
        "    tensor<fp16,[1,%d,%d]> tail=mul(x=xoT2,y=sc)[name=tensor<string,[]>(\"tail\")];\n"
        "    tensor<fp16,[1,%d,%d]> htail=mul(x=half,y=tail)[name=tensor<string,[]>(\"htail\")];\n"
        "    tensor<fp16,[1,%d,%d]> gpx =add(x=hTp1,y=htail)[name=tensor<string,[]>(\"gpx\")];\n"
        "    tensor<fp16,[1,%d,%d]> dx  =mul(x=dy, y=gpx)[name=tensor<string,[]>(\"dx\")];\n"
        "  } -> (dx);\n}\n",
        C,S, C,S,
        C,S, C,S, C,S, C,S, C,S, C,S,
        C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S, C,S];
    size_t ins[2] = {sn, sn};
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins, 1, &sn);
    if (!k) { fprintf(stderr, "ane_gelu_bwd_compile FAILED (C=%d S=%d)\n", C, S); return NULL; }
    ANEGeluBwd *g = (ANEGeluBwd *)calloc(1, sizeof(ANEGeluBwd));
    g->C = C; g->S = S; g->k = k;
    g->dy_surf = k->ioInputs[0];
    g->x_surf  = k->ioInputs[1];
    g->dx_surf = k->ioOutputs[0];
    return g;
}

#ifndef _GELU_CPU_THRESHOLD
#define _GELU_CPU_THRESHOLD (50*1024)
#endif
// CPU NEON GELU backward
static void _ane_gelu_bwd_eval_cpu(ANEGeluBwd *g) {
    int N = g->C * g->S;
    IOSurfaceLock(g->dy_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(g->x_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(g->dx_surf, 0, NULL);
    const _Float16 *dy = (const _Float16 *)IOSurfaceGetBaseAddress(g->dy_surf);
    const _Float16 *x  = (const _Float16 *)IOSurfaceGetBaseAddress(g->x_surf);
    _Float16       *dx = (_Float16 *)IOSurfaceGetBaseAddress(g->dx_surf);
    const float c0 = 0.7978845608f, c1 = 0.044715f;
    for (int i = 0; i < N; i++) {
        float xi = (float)x[i];
        float arg = c0 * (xi + c1 * xi * xi * xi);
        float T = tanhf(arg);
        float gpx = 0.5f * (1.0f + T) + 0.5f * xi * (1.0f - T*T) * c0 * (1.0f + 3.0f * c1 * xi * xi);
        dx[i] = (_Float16)((float)dy[i] * gpx);
    }
    IOSurfaceUnlock(g->dx_surf, 0, NULL);
    IOSurfaceUnlock(g->x_surf,  kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(g->dy_surf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_gelu_bwd_eval(ANEGeluBwd *g) { ane_eval(g->k); }

static void ane_gelu_bwd_write_dy(ANEGeluBwd *g, const _Float16 *dy) {
    IOSurfaceLock(g->dy_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(g->dy_surf), dy, g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->dy_surf, 0, NULL);
}

static void ane_gelu_bwd_write_x(ANEGeluBwd *g, const _Float16 *x) {
    IOSurfaceLock(g->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(g->x_surf), x, g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->x_surf, 0, NULL);
}

static void ane_gelu_bwd_read_dx(ANEGeluBwd *g, _Float16 *dx) {
    IOSurfaceLock(g->dx_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(dx, IOSurfaceGetBaseAddress(g->dx_surf), g->C * g->S * sizeof(_Float16));
    IOSurfaceUnlock(g->dx_surf, kIOSurfaceLockReadOnly, NULL);
}
