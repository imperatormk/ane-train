// ane_fuse_bwd.h — Fuse backward: matmul bwd + channel-split for skip/up dx
//
// Forward: concat(skip[Cskip,S], up[Cup,S]) → cat[Cin,S] → W[Cout,Cin] @ cat → y[Cout,S]
// Backward:
//   dW[Cout,Cin] = dy @ cat^T         (ANE matmul_bwd dW)
//   d_cat[Cin,S] = W^T @ dy           (ANE matmul_bwd dX)
//   d_skip = d_cat[0:Cskip, :]
//   d_up   = d_cat[Cskip:Cin, :]
//
#pragma once
#include "ane_fuse.h"
#include "ane_matmul_bwd.h"

typedef struct {
    ANEFuse      *fwd;
    ANEMatmulBwd *mm_bwd;
    int Cskip, Cup;
    _Float16 *d_cat;   // [Cin, S] scratch
} ANEFuseBwd;

static ANEFuseBwd *ane_fuse_bwd_compile(ANEFuse *fwd, int Cskip, int Cup) {
    ANEMatmulBwd *mm_bwd = ane_matmul_bwd_compile(fwd->Cin, fwd->Cout, fwd->S);
    if (!mm_bwd) { fprintf(stderr, "ane_fuse_bwd_compile FAILED\n"); return NULL; }
    ane_matmul_bwd_rewire_w(mm_bwd, fwd->mm->w_surf);
    // x (cat_buf) is written to mm->x_surf each fwd pass — share it
    ane_matmul_bwd_rewire_x(mm_bwd, fwd->mm->x_surf);

    ANEFuseBwd *bwd = (ANEFuseBwd *)calloc(1, sizeof(ANEFuseBwd));
    bwd->fwd    = fwd;
    bwd->mm_bwd = mm_bwd;
    bwd->Cskip  = Cskip;
    bwd->Cup    = Cup;
    bwd->d_cat  = (_Float16 *)malloc((size_t)fwd->Cin * fwd->S * sizeof(_Float16));
    return bwd;
}

// dy[Cout,S] → d_skip[Cskip,S], d_up[Cup,S]
// Read dx_surf once and split directly — avoids d_cat intermediate + 3 memcpy passes.
static void ane_fuse_bwd_eval(ANEFuseBwd *bwd, const _Float16 *dy,
                               _Float16 *d_skip, _Float16 *d_up) {
    int S = bwd->fwd->S;
    ane_matmul_bwd_write_dy(bwd->mm_bwd, dy);
    ane_matmul_bwd_eval(bwd->mm_bwd);
    // Lock dx_surf once, split directly into d_skip and d_up
    IOSurfaceLock(bwd->mm_bwd->dx_surf, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *d_cat = (const _Float16 *)IOSurfaceGetBaseAddress(bwd->mm_bwd->dx_surf);
    memcpy(d_skip, d_cat,                   (size_t)bwd->Cskip * S * sizeof(_Float16));
    memcpy(d_up,   d_cat + bwd->Cskip * S,  (size_t)bwd->Cup   * S * sizeof(_Float16));
    IOSurfaceUnlock(bwd->mm_bwd->dx_surf, kIOSurfaceLockReadOnly, NULL);
}

static void ane_fuse_bwd_read_dw(ANEFuseBwd *bwd, _Float16 *dw) {
    ane_matmul_bwd_read_dw(bwd->mm_bwd, dw);
}
