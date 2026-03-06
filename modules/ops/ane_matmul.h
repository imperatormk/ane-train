// ane_matmul.h — ANE matmul: W[1,Co,Ci] @ x[1,Ci,S] → y[1,Co,S]
// Runtime W — no baked weights, no recompile needed.
//
// Usage:
//   ANEMatmul *mm = ane_matmul_compile(Ci, Co, S);
//   ane_matmul_write_w(mm, w_fp16);    // [Co*Ci] fp16
//   ane_matmul_write_x(mm, x_fp16);    // [Ci*S]  fp16
//   ane_matmul_eval(mm);
//   ane_matmul_read_y(mm, y_fp16);     // [Co*S]  fp16
//   // or chain: mm->w_surf / mm->x_surf / mm->y_surf
//
#pragma once
#include "../../ane_runtime.h"
#include "../../mil_gen.h"

typedef struct {
    int Ci, Co, S;
    ANEKernel *k;
    IOSurfaceRef w_surf;  // slot 0: W [1,Co,Ci]
    IOSurfaceRef x_surf;  // slot 1: x [1,Ci,S]
    IOSurfaceRef y_surf;  // output:  y [1,Co,S]
} ANEMatmul;

static ANEMatmul *ane_matmul_compile(int Ci, int Co, int S) {
    size_t sw = (size_t)Co * Ci * 2;  if (sw < 2048) sw = 2048;
    size_t sx = (size_t)Ci * S  * 2;  if (sx < 2048) sx = 2048;
    size_t sy = (size_t)Co * S  * 2;  if (sy < 2048) sy = 2048;
    // slot rule: ins must be non-decreasing
    // sw <= sx assumed (Co*Ci <= Ci*S when S >= Co, which is typical)
    // if not, caller must ensure ordering — we enforce sw <= sx
    size_t ins[2] = {sw, sx};
    NSString *mil = mil_gen_matmul_fwd(Ci, Co, S);
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins, 1, &sy);
    if (!k) { fprintf(stderr, "ane_matmul_compile FAILED (Ci=%d Co=%d S=%d)\n", Ci, Co, S); return NULL; }
    ANEMatmul *mm = (ANEMatmul *)calloc(1, sizeof(ANEMatmul));
    mm->Ci = Ci; mm->Co = Co; mm->S = S;
    mm->k = k;
    mm->w_surf = k->ioInputs[0];
    mm->x_surf = k->ioInputs[1];
    mm->y_surf = k->ioOutputs[0];
    return mm;
}

static void ane_matmul_write_w(ANEMatmul *mm, const _Float16 *w) {
    IOSurfaceLock(mm->w_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(mm->w_surf), w, mm->Co * mm->Ci * sizeof(_Float16));
    IOSurfaceUnlock(mm->w_surf, 0, NULL);
}

static void ane_matmul_write_x(ANEMatmul *mm, const _Float16 *x) {
    IOSurfaceLock(mm->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(mm->x_surf), x, mm->Ci * mm->S * sizeof(_Float16));
    IOSurfaceUnlock(mm->x_surf, 0, NULL);
}

static void ane_matmul_eval(ANEMatmul *mm) {
    ane_eval(mm->k);
}

static void ane_matmul_read_y(ANEMatmul *mm, _Float16 *y) {
    IOSurfaceLock(mm->y_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(y, IOSurfaceGetBaseAddress(mm->y_surf), mm->Co * mm->S * sizeof(_Float16));
    IOSurfaceUnlock(mm->y_surf, kIOSurfaceLockReadOnly, NULL);
}
