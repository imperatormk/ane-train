// ane_add.h — ANE element-wise add: a[1,C,S] + b[1,C,S] → c[1,C,S]
// Used for residual connections.
//
// Usage:
//   ANEAdd *add = ane_add_compile(C, S);
//   // wire a_surf / b_surf via ane_rewire, or write directly
//   ane_add_eval(add);
//   // read c_surf or chain into next kernel
//
#pragma once
#include "../../ane_runtime.h"
#include "../../mil_gen.h"

typedef struct {
    int C, S;
    ANEKernel *k;
    IOSurfaceRef a_surf;
    IOSurfaceRef b_surf;
    IOSurfaceRef c_surf;
} ANEAdd;

static void _ane_add_rewire_in(ANEKernel *k, int slot, IOSurfaceRef surf) {
    IOSurfaceRef ins[4] = {0};
    ins[slot] = surf;
    ane_rewire(k, ins, NULL);
}

static ANEAdd *ane_add_compile(int C, int S) {
    size_t sz = (size_t)C * S * 2;  if (sz < 2048) sz = 2048;
    size_t ins[2] = {sz, sz};
    NSString *mil = mil_gen_add(C, S);
    ANEKernel *k = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], nil, 2, ins, 1, &sz);
    if (!k) { fprintf(stderr, "ane_add_compile FAILED (C=%d S=%d)\n", C, S); return NULL; }
    ANEAdd *a = (ANEAdd *)calloc(1, sizeof(ANEAdd));
    a->C = C; a->S = S;
    a->k = k;
    a->a_surf = k->ioInputs[0];
    a->b_surf = k->ioInputs[1];
    a->c_surf = k->ioOutputs[0];
    return a;
}

// Rewire input slots (for chaining)
static void ane_add_rewire_a(ANEAdd *a, IOSurfaceRef surf) { _ane_add_rewire_in(a->k, 0, surf); a->a_surf = surf; }
static void ane_add_rewire_b(ANEAdd *a, IOSurfaceRef surf) { _ane_add_rewire_in(a->k, 1, surf); a->b_surf = surf; }

static void ane_add_eval(ANEAdd *a) { ane_eval(a->k); }

static void ane_add_read_c(ANEAdd *a, _Float16 *c) {
    IOSurfaceLock(a->c_surf, kIOSurfaceLockReadOnly, NULL);
    memcpy(c, IOSurfaceGetBaseAddress(a->c_surf), a->C * a->S * sizeof(_Float16));
    IOSurfaceUnlock(a->c_surf, kIOSurfaceLockReadOnly, NULL);
}
