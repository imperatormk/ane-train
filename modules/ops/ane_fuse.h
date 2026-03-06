// ane_fuse.h — Fuse: 1×1 conv (matmul) to project concatenated skip+up features
//
// After upsample + concat, fuse reduces channels:
//   [Cin, S] → [Cout, S]  via W[Cout, Cin] @ x[Cin, S]
//
// Cin is typically Cskip + Cup (e.g. 192+384=576 → 192, or 96+192=288 → 96).
// Concat is done on CPU (two memcpy into col_buf), then ANE matmul.
//
// Usage:
//   ANEFuse *fuse = ane_fuse_compile(Cin, Cout, S);
//   ane_fuse_write_w(fuse, w_fp16);              // [Cout, Cin]
//   ane_fuse_eval(fuse, skip_fp16, up_fp16,      // [Cskip,S] + [Cup,S]
//                 Cskip, Cup);
//   ane_fuse_read_y(fuse, y_fp16);               // [Cout, S]
//
#pragma once
#include "../../ane_runtime.h"
#include "ane_matmul.h"

typedef struct {
    int Cin, Cout, S;
    ANEMatmul *mm;
    _Float16 *cat_buf;  // [Cin, S] scratch for concat
} ANEFuse;

static ANEFuse *ane_fuse_compile(int Cin, int Cout, int S) {
    ANEMatmul *mm = ane_matmul_compile(Cin, Cout, S);
    if (!mm) {
        fprintf(stderr, "ane_fuse_compile FAILED (Cin=%d Cout=%d S=%d)\n", Cin, Cout, S);
        return NULL;
    }
    ANEFuse *fuse = (ANEFuse *)calloc(1, sizeof(ANEFuse));
    fuse->Cin = Cin; fuse->Cout = Cout; fuse->S = S;
    fuse->mm = mm;
    fuse->cat_buf = (_Float16 *)malloc((size_t)Cin * S * sizeof(_Float16));
    return fuse;
}

static void ane_fuse_write_w(ANEFuse *fuse, const _Float16 *w) {
    ane_matmul_write_w(fuse->mm, w);
}

// Concatenate skip[Cskip, S] and up[Cup, S] into cat_buf[Cin, S], then run matmul
static void ane_fuse_eval(ANEFuse *fuse, const _Float16 *skip, const _Float16 *up,
                           int Cskip, int Cup) {
    int S = fuse->S;
    memcpy(fuse->cat_buf,           skip, (size_t)Cskip * S * sizeof(_Float16));
    memcpy(fuse->cat_buf + Cskip*S, up,   (size_t)Cup   * S * sizeof(_Float16));
    IOSurfaceLock(fuse->mm->x_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(fuse->mm->x_surf), fuse->cat_buf,
           (size_t)fuse->Cin * S * sizeof(_Float16));
    IOSurfaceUnlock(fuse->mm->x_surf, 0, NULL);
    ane_matmul_eval(fuse->mm);
}

static void ane_fuse_read_y(ANEFuse *fuse, _Float16 *y) {
    ane_matmul_read_y(fuse->mm, y);
}

static IOSurfaceRef ane_fuse_output_surf(ANEFuse *fuse) { return fuse->mm->y_surf; }
