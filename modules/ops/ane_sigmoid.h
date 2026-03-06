// ane_sigmoid.h — ANE sigmoid: y = 1/(1+exp(-x))
// Layout: [1, C, S]
//
// Usage:
//   ANESigmoid *sg = ane_sigmoid_compile(C, S);
//   ane_sigmoid_write_x(sg, x_fp16);
//   ane_sigmoid_eval(sg);
//   ane_sigmoid_read_y(sg, y_fp16);
//
#pragma once
#include "../../ane_runtime.h"

#define ANE_SIG_BI "[buildInfo=dict<tensor<string,[]>,tensor<string,[]>>({{\"coremlc-version\",\"3505.4.1\"}})]\n"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef x_surf;
    IOSurfaceRef y_surf;
} ANESigmoid;

static ANESigmoid *ane_sigmoid_compile(int C, int S) {
    size_t sz = (size_t)C * S * 2;  if (sz < 2048) sz = 2048;
    NSString *mil = [NSString stringWithFormat:
        @"program(1.0)\n" ANE_SIG_BI
        "{\n  func main<ios16>(tensor<fp16,[1,%d,%d]> X) {\n"
        "    tensor<fp16,[1,%d,%d]> Y=sigmoid(x=X)[name=tensor<string,[]>(\"Y\")];\n"
        "  } -> (Y);\n}\n",
        C, S, C, S];
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding],
                                nil, 1, &sz, 1, &sz);
    if (!k) { fprintf(stderr, "ane_sigmoid_compile FAILED (C=%d S=%d)\n", C, S); return NULL; }
    ANESigmoid *sg = (ANESigmoid *)calloc(1, sizeof(ANESigmoid));
    sg->C = C; sg->S = S;
    sg->k = k;
    sg->x_surf = k->ioInputs[0];
    sg->y_surf = k->ioOutputs[0];
    return sg;
}

static void ane_sigmoid_write_x(ANESigmoid *sg, const _Float16 *x) {
    IOSurfaceLock(sg->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(sg->x_surf), x, sg->C * sg->S * sizeof(_Float16));
    IOSurfaceUnlock(sg->x_surf, 0, NULL);
}

static void ane_sigmoid_eval(ANESigmoid *sg) { ane_eval(sg->k); }

static void ane_sigmoid_read_y(ANESigmoid *sg, _Float16 *y) {
    IOSurfaceLock(sg->y_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(sg->y_surf), sg->C * sg->S * sizeof(_Float16));
    IOSurfaceUnlock(sg->y_surf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_sigmoid_rewire_x(ANESigmoid *sg, IOSurfaceRef surf) {
    IOSurfaceRef ins[2] = {surf, 0};
    ane_rewire(sg->k, ins, NULL);
    sg->x_surf = surf;
}
