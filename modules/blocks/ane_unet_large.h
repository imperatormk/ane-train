// ane_unet_large.h — ~20M param ANE ConvNeXt UNet encoder/decoder
//
// Architecture (no Swin/MoE/ASPP — pure ConvNeXt + global attention):
//   stem:   4×4 s2,  3→96  @ H/2     (S1)
//   stage1: nB1 × ConvNeXt(96)        (ref: 4)
//   down1:  96→192  @ H/4             (S2)
//   stage2: nB2 × ConvNeXt(192)       (ref: 6)
//   down2:  192→384 @ H/8             (S3)
//   stage3: nB3 × ConvNeXt(384)       (ref: 10) ← most params
//   attn:   nA  × GlobalAttn(384,S3)  (ref: 6)
//   up1+fuse1: (384 up + skip2[192])→192 @ S2
//   stage4: nB4 × ConvNeXt(192)       (ref: 4)
//   up2+fuse2: (192 up + skip1[96])→96  @ S1
//   stage5: nB5 × ConvNeXt(96)        (ref: 4)
//   head:   96→1 @ S0
//
// Param count (nB1=4,nB2=6,nB3=10,nB4=4,nB5=4,nA=6): ~20M
//
// Usage:
//   ANEUNetLarge *net = ane_unet_large_compile(H, nB1,nB2,nB3,nB4,nB5, nA);
//   ane_unet_large_set_weights(net, &w);
//   ANEUNetLargeBwd *bwd = ane_unet_large_bwd_compile(net, lr, b1, b2, eps, /*checkpointed=*/1);
//
#pragma once
#include "ane_convnext.h"
#include "ane_convnext_bwd.h"
#include "../ops/ane_attn.h"
#include "../ops/ane_attn_bwd.h"
#include "../ops/ane_stem.h"
#include "../ops/ane_stem_bwd.h"
#include "../ops/ane_down.h"
#include "../ops/ane_down_bwd.h"
#include "../ops/ane_fuse.h"
#include "../ops/ane_fuse_bwd.h"
#include "../ops/ane_matmul.h"
#include "../ops/ane_matmul_bwd.h"
#include "../ops/ane_upsample2x.h"
#include "../ops/ane_upsample2x_bwd.h"
#include "../ops/ane_adam.h"
#include "../ops/ane_loss.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ---- Forward model ----

typedef struct {
    int H, S0, S1, S2, S3;
    int nB1, nB2, nB3, nB4, nB5, nA;

    ANEStem       *stem;
    ANEConvNeXt  **stage1;  // [nB1]
    ANEDown       *down1;
    ANEConvNeXt  **stage2;  // [nB2]
    ANEDown       *down2;
    ANEConvNeXt  **stage3;  // [nB3]
    ANEAttn      **attn;    // [nA]
    ANEFuse       *fuse1;   // (384+192)→192
    ANEConvNeXt  **stage4;  // [nB4]
    ANEFuse       *fuse2;   // (192+96)→96
    ANEConvNeXt  **stage5;  // [nB5]
    ANEMatmul     *head;    // 96→1

    // Scratch buffers
    _Float16 *skip1;      // [96,  S1]
    _Float16 *skip2;      // [192, S2]
    _Float16 *stage_buf;  // [384, S2] (reused for stage1-3 intermediates)
    _Float16 *up_buf;     // [384, S3] stage3 output + attn I/O
    _Float16 *ups1_buf;   // [384, S2] upsample stage3→fuse1 (384ch, S3→S2)
    _Float16 *ups2_buf;   // [192, S1] upsample stage4→fuse2 (192ch, S2→S1)
    _Float16 *up3_buf;    // [96,  S0] upsample stage5→head  (96ch,  S1→S0)
    _Float16 *fuse1_out;  // [192, S2]
    _Float16 *fuse2_out;  // [96,  S1]
} ANEUNetLarge;

static ANEUNetLarge *ane_unet_large_compile(int H,
                                             int nB1, int nB2, int nB3, int nB4, int nB5,
                                             int nA) {
    ANEUNetLarge *net = (ANEUNetLarge *)calloc(1, sizeof(ANEUNetLarge));
    net->H = H;
    net->nB1=nB1; net->nB2=nB2; net->nB3=nB3; net->nB4=nB4; net->nB5=nB5; net->nA=nA;
    net->S0 = H*H;
    net->S1 = (H/2)*(H/2);
    net->S2 = (H/4)*(H/4);
    net->S3 = (H/8)*(H/8);
    int S1=net->S1, S2=net->S2, S3=net->S3, S0=net->S0;

    printf("[UNetLarge] H=%d nB=(%d,%d,%d,%d,%d) nA=%d  S=(%d,%d,%d,%d)\n",
           H, nB1,nB2,nB3,nB4,nB5, nA, S0,S1,S2,S3);

#define CHK(x,l) if(!(x)){fprintf(stderr,"ane_unet_large_compile FAILED: %s\n",l);return NULL;}

    net->stem = ane_stem_compile(3, 96, H, 4); CHK(net->stem,"stem");

#define COMPILE_STAGE(arr, nb, C, S, Hb, label) do { \
    arr = (ANEConvNeXt **)calloc(nb, sizeof(void*)); \
    for (int i=0;i<nb;i++) { \
        arr[i] = ane_convnext_compile(C, S, 7); CHK(arr[i], label); \
        if (i>0) ane_convnext_chain(arr[i-1], arr[i]); \
    } \
} while(0)

    COMPILE_STAGE(net->stage1, nB1, 96,  S1, H/2, "stage1");
    net->down1 = ane_down_compile(96, 192, H/2);   CHK(net->down1,"down1");
    COMPILE_STAGE(net->stage2, nB2, 192, S2, H/4, "stage2");
    net->down2 = ane_down_compile(192, 384, H/4);  CHK(net->down2,"down2");
    COMPILE_STAGE(net->stage3, nB3, 384, S3, H/8, "stage3");

    net->attn = (ANEAttn **)calloc(nA, sizeof(void*));
    for (int i=0;i<nA;i++) { net->attn[i]=ane_attn_compile(384,S3); CHK(net->attn[i],"attn"); }

    net->fuse1 = ane_fuse_compile(576, 192, S2); CHK(net->fuse1,"fuse1");
    COMPILE_STAGE(net->stage4, nB4, 192, S2, H/4, "stage4");
    net->fuse2 = ane_fuse_compile(288, 96, S1);  CHK(net->fuse2,"fuse2");
    COMPILE_STAGE(net->stage5, nB5, 96,  S1, H/2, "stage5");

    net->head = ane_matmul_compile(96, 1, S0); CHK(net->head,"head");

#undef COMPILE_STAGE
#undef CHK

    net->skip1     = (_Float16 *)malloc((size_t)96  * S1 * 2);
    net->skip2     = (_Float16 *)malloc((size_t)192 * S2 * 2);
    net->stage_buf = (_Float16 *)malloc((size_t)384 * S2 * 2);
    net->up_buf    = (_Float16 *)malloc((size_t)384 * S3 * 2);  // stage3 + attn
    net->ups1_buf  = (_Float16 *)malloc((size_t)384 * S2 * 2);  // 384ch, S3→S2
    net->ups2_buf  = (_Float16 *)malloc((size_t)192 * S1 * 2);  // 192ch, S2→S1
    net->up3_buf   = (_Float16 *)malloc((size_t)96  * S0 * 2);  // 96ch,  S1→S0
    net->fuse1_out = (_Float16 *)malloc((size_t)192 * S2 * 2);
    net->fuse2_out = (_Float16 *)malloc((size_t)96  * S1 * 2);

    printf("[UNetLarge] compiled OK\n");
    return net;
}

// ---- Weights struct ----

typedef struct {
    _Float16  *stem;
    _Float16 **stage1_dw, **stage1_pw1, **stage1_pw2;  // [nB1]
    _Float16 **stage2_dw, **stage2_pw1, **stage2_pw2;  // [nB2]
    _Float16 **stage3_dw, **stage3_pw1, **stage3_pw2;  // [nB3]
    _Float16 **stage4_dw, **stage4_pw1, **stage4_pw2;  // [nB4]
    _Float16 **stage5_dw, **stage5_pw1, **stage5_pw2;  // [nB5]
    _Float16  *down1, *down2;
    _Float16 **attn_Wq, **attn_Wk, **attn_Wv, **attn_Wo;  // [nA]
    _Float16  *fuse1, *fuse2, *head;
} ANEUNetLargeWeights;

static uint32_t _lrng = 0xDEADBEEF;
static float _lrandf(void) {
    _lrng ^= _lrng<<13; _lrng ^= _lrng>>17; _lrng ^= _lrng<<5;
    return (_lrng & 0xFFFFFF) / (float)0x1000000;
}
static void _lkaiming(_Float16 *w, int n, int fan_in) {
    float s = sqrtf(1.0f/fan_in);
    for (int i=0;i<n;i++) w[i]=(_Float16)((_lrandf()*2.f-1.f)*s);
}

#define _LALLOC_STAGE(arr_dw, arr_pw1, arr_pw2, nb, C) do { \
    arr_dw  = calloc(nb,8); arr_pw1 = calloc(nb,8); arr_pw2 = calloc(nb,8); \
    for (int i=0;i<nb;i++) { \
        arr_dw[i]  = malloc((size_t)C*49*2);     _lkaiming(arr_dw[i],  C*49,  49); \
        arr_pw1[i] = malloc((size_t)C*4*C*2);    _lkaiming(arr_pw1[i], C*4*C, C); \
        arr_pw2[i] = malloc((size_t)C*4*C*2);    _lkaiming(arr_pw2[i], C*4*C, C*4); \
    } \
} while(0)

static ANEUNetLargeWeights ane_unet_large_make_weights(int nB1,int nB2,int nB3,int nB4,int nB5,int nA) {
    ANEUNetLargeWeights w = {0};
    w.stem = malloc(96*64*2); _lkaiming(w.stem, 96*64, 3*4*4);
    _LALLOC_STAGE(w.stage1_dw,w.stage1_pw1,w.stage1_pw2, nB1, 96);
    _LALLOC_STAGE(w.stage2_dw,w.stage2_pw1,w.stage2_pw2, nB2, 192);
    _LALLOC_STAGE(w.stage3_dw,w.stage3_pw1,w.stage3_pw2, nB3, 384);
    _LALLOC_STAGE(w.stage4_dw,w.stage4_pw1,w.stage4_pw2, nB4, 192);
    _LALLOC_STAGE(w.stage5_dw,w.stage5_pw1,w.stage5_pw2, nB5, 96);
    w.down1 = malloc(192*384*2); _lkaiming(w.down1, 192*384, 4*96);
    w.down2 = malloc(384*768*2); _lkaiming(w.down2, 384*768, 4*192);
    w.attn_Wq=calloc(nA,8); w.attn_Wk=calloc(nA,8); w.attn_Wv=calloc(nA,8); w.attn_Wo=calloc(nA,8);
    for (int i=0;i<nA;i++) {
        w.attn_Wq[i]=malloc(384*384*2); _lkaiming(w.attn_Wq[i],384*384,384);
        w.attn_Wk[i]=malloc(384*384*2); _lkaiming(w.attn_Wk[i],384*384,384);
        w.attn_Wv[i]=malloc(384*384*2); _lkaiming(w.attn_Wv[i],384*384,384);
        w.attn_Wo[i]=malloc(384*384*2); _lkaiming(w.attn_Wo[i],384*384,384);
    }
    w.fuse1 = malloc(192*576*2); _lkaiming(w.fuse1, 192*576, 576);
    w.fuse2 = malloc(96*288*2);  _lkaiming(w.fuse2, 96*288,  288);
    w.head  = calloc(96, 2);
    return w;
}
#undef _LALLOC_STAGE

static void ane_unet_large_set_weights(ANEUNetLarge *net, const ANEUNetLargeWeights *w) {
    ane_stem_write_w(net->stem, w->stem);
    for (int i=0;i<net->nB1;i++) ane_convnext_set_weights(net->stage1[i],w->stage1_dw[i],w->stage1_pw1[i],w->stage1_pw2[i]);
    ane_down_write_w(net->down1, w->down1);
    for (int i=0;i<net->nB2;i++) ane_convnext_set_weights(net->stage2[i],w->stage2_dw[i],w->stage2_pw1[i],w->stage2_pw2[i]);
    ane_down_write_w(net->down2, w->down2);
    for (int i=0;i<net->nB3;i++) ane_convnext_set_weights(net->stage3[i],w->stage3_dw[i],w->stage3_pw1[i],w->stage3_pw2[i]);
    for (int i=0;i<net->nA;i++)  ane_attn_set_weights(net->attn[i],w->attn_Wq[i],w->attn_Wk[i],w->attn_Wv[i],w->attn_Wo[i]);
    ane_fuse_write_w(net->fuse1, w->fuse1);
    for (int i=0;i<net->nB4;i++) ane_convnext_set_weights(net->stage4[i],w->stage4_dw[i],w->stage4_pw1[i],w->stage4_pw2[i]);
    ane_fuse_write_w(net->fuse2, w->fuse2);
    for (int i=0;i<net->nB5;i++) ane_convnext_set_weights(net->stage5[i],w->stage5_dw[i],w->stage5_pw1[i],w->stage5_pw2[i]);
    ane_matmul_write_w(net->head, w->head);
}

// ---- Forward pass ----

static void _large_run_stage(ANEConvNeXt **blocks, int nB, int H,
                              const _Float16 *in, _Float16 *out) {
    ane_convnext_write_input(blocks[0], in);
    for (int i=0;i<nB;i++) ane_convnext_eval(blocks[i], H);
    ane_convnext_read_output(blocks[nB-1], out);
}

static int _unet_large_fwd_step = 0;
static double _unet_large_now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e3+ts.tv_nsec*1e-6;}

static void ane_unet_large_eval(ANEUNetLarge *net, const _Float16 *x_rgb) {
    int H=net->H, S0=net->S0, S1=net->S1, S2=net->S2, S3=net->S3;
    int _p = (++_unet_large_fwd_step == 2);
    double _t0=0, _t1;
#define _FT(label) if(_p){_t1=_unet_large_now_ms();printf("  fwd %-20s %.1f ms\n",label,_t1-_t0);_t0=_t1;}
    if(_p) _t0=_unet_large_now_ms();

    ane_stem_eval(net->stem, x_rgb);
    ane_stem_read_y(net->stem, net->skip1);
    _FT("stem")

    _large_run_stage(net->stage1, net->nB1, H/2, net->skip1, net->skip1);
    _FT("stage1(4blk,96ch,128²)")

    ane_down_eval(net->down1, net->skip1);
    ane_down_read_y(net->down1, net->stage_buf);
    _large_run_stage(net->stage2, net->nB2, H/4, net->stage_buf, net->skip2);
    _FT("down1+stage2(6blk,192ch,64²)")

    ane_down_eval(net->down2, net->skip2);
    ane_down_read_y(net->down2, net->stage_buf);
    _large_run_stage(net->stage3, net->nB3, H/8, net->stage_buf, net->up_buf);
    _FT("down2+stage3(10blk,384ch,32²)")

    for (int i=0;i<net->nA;i++) {
        ane_attn_write_x(net->attn[i], net->up_buf);
        ane_attn_eval(net->attn[i]);
        ane_attn_read_y(net->attn[i], net->up_buf);
    }
    _FT("attn(6×,384ch,32²)")

    ane_upsample2x(net->up_buf, net->ups1_buf, 384, H/8);
    ane_fuse_eval(net->fuse1, net->skip2, net->ups1_buf, 192, 384);
    ane_fuse_read_y(net->fuse1, net->fuse1_out);
    _large_run_stage(net->stage4, net->nB4, H/4, net->fuse1_out, net->fuse2_out);
    _FT("up1+fuse1+stage4(4blk,192ch,64²)")

    ane_upsample2x(net->fuse2_out, net->ups2_buf, 192, H/4);
    ane_fuse_eval(net->fuse2, net->skip1, net->ups2_buf, 96, 192);
    ane_fuse_read_y(net->fuse2, net->fuse2_out);
    _large_run_stage(net->stage5, net->nB5, H/2, net->fuse2_out, net->stage_buf);
    _FT("up2+fuse2+stage5(4blk,96ch,128²)")

    ane_upsample2x(net->stage_buf, net->up3_buf, 96, H/2);
    ane_matmul_write_x(net->head, net->up3_buf);
    ane_matmul_eval(net->head);
    _FT("up3+head")
#undef _FT
}

static void ane_unet_large_read_output(ANEUNetLarge *net, _Float16 *out) {
    IOSurfaceLock(net->head->y_surf, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *y = (const _Float16 *)IOSurfaceGetBaseAddress(net->head->y_surf);
    int S0 = net->S0;
    for (int i=0;i<S0;i++) out[i] = (_Float16)(1.0f/(1.0f+expf(-(float)y[i])));
    IOSurfaceUnlock(net->head->y_surf, kIOSurfaceLockReadOnly, NULL);
}

// ---- Backward model ----

typedef struct {
    int H, nB1, nB2, nB3, nB4, nB5, nA;

    ANEConvNeXtBwd **stage1_bwd;
    ANEConvNeXtBwd **stage2_bwd;
    ANEConvNeXtBwd **stage3_bwd;
    ANEConvNeXtBwd **stage4_bwd;
    ANEConvNeXtBwd **stage5_bwd;
    ANEAttnBwd     **attn_bwd;
    ANEStemBwd      *stem_bwd;
    ANEDownBwd      *down1_bwd, *down2_bwd;
    ANEFuseBwd      *fuse1_bwd, *fuse2_bwd;
    ANEMatmulBwd    *head_bwd;

    // Adam optimizers
    ANEAdam *opt_stem, *opt_down1, *opt_down2, *opt_fuse1, *opt_fuse2, *opt_head;
    ANEAdam **opt_attn_Wq, **opt_attn_Wk, **opt_attn_Wv, **opt_attn_Wo;

    // Gradient scratch
    _Float16 *dy_head, *dx_head;
    _Float16 *dx_up3, *dy_stage5, *dx_stage5;
    _Float16 *d_skip1, *d_up2, *dx_up2;
    _Float16 *dy_stage4, *dx_stage4;
    _Float16 *d_skip2, *d_up1, *dx_up1;
    _Float16 *dy_attn, *dx_attn;
    _Float16 *dy_stage3, *dx_stage3;
    _Float16 *dy_stage2, *dx_stage2;
    _Float16 *dy_stage1, *dx_stage1, *dx_stem;
    _Float16 *tmp2, *tmp3;  // scratch for inter-block grads

    // dW scratch
    _Float16 *dw_stem, *dw_down1, *dw_down2, *dw_fuse1, *dw_fuse2, *dw_head;
    _Float16 **dw_attn_Wq, **dw_attn_Wk, **dw_attn_Wv, **dw_attn_Wo;
} ANEUNetLargeBwd;

static ANEUNetLargeBwd *ane_unet_large_bwd_compile(ANEUNetLarge *net,
                                                    float lr, float b1, float b2, float eps,
                                                    int checkpointed) {
    int H=net->H, nB1=net->nB1, nB2=net->nB2, nB3=net->nB3, nB4=net->nB4, nB5=net->nB5, nA=net->nA;
    int S0=net->S0, S1=net->S1, S2=net->S2, S3=net->S3;
    ANEUNetLargeBwd *bwd = (ANEUNetLargeBwd *)calloc(1, sizeof(ANEUNetLargeBwd));
    bwd->H=H; bwd->nB1=nB1; bwd->nB2=nB2; bwd->nB3=nB3; bwd->nB4=nB4; bwd->nB5=nB5; bwd->nA=nA;

#define CHK(x,l) if(!(x)){fprintf(stderr,"ane_unet_large_bwd_compile FAILED: %s\n",l);return NULL;}
#define COMPILE_STAGE_BWD(arr, net_arr, nb) do { \
    arr = (ANEConvNeXtBwd **)calloc(nb, sizeof(void*)); \
    for (int i=0;i<nb;i++) { \
        arr[i] = ane_convnext_bwd_compile_ex(net_arr[i], lr, b1, b2, eps, checkpointed); \
        CHK(arr[i], #arr); \
    } \
} while(0)

    COMPILE_STAGE_BWD(bwd->stage1_bwd, net->stage1, nB1);
    COMPILE_STAGE_BWD(bwd->stage2_bwd, net->stage2, nB2);
    COMPILE_STAGE_BWD(bwd->stage3_bwd, net->stage3, nB3);
    COMPILE_STAGE_BWD(bwd->stage4_bwd, net->stage4, nB4);
    COMPILE_STAGE_BWD(bwd->stage5_bwd, net->stage5, nB5);
#undef COMPILE_STAGE_BWD

    bwd->attn_bwd = (ANEAttnBwd **)calloc(nA, sizeof(void*));
    for (int i=0;i<nA;i++) { bwd->attn_bwd[i]=ane_attn_bwd_compile(net->attn[i]); CHK(bwd->attn_bwd[i],"attn_bwd"); }

    bwd->stem_bwd  = ane_stem_bwd_compile(net->stem);        CHK(bwd->stem_bwd,"stem_bwd");
    bwd->down1_bwd = ane_down_bwd_compile(net->down1);       CHK(bwd->down1_bwd,"down1_bwd");
    bwd->down2_bwd = ane_down_bwd_compile(net->down2);       CHK(bwd->down2_bwd,"down2_bwd");
    bwd->fuse1_bwd = ane_fuse_bwd_compile(net->fuse1,192,384); CHK(bwd->fuse1_bwd,"fuse1_bwd");
    bwd->fuse2_bwd = ane_fuse_bwd_compile(net->fuse2,96,192);  CHK(bwd->fuse2_bwd,"fuse2_bwd");
    bwd->head_bwd  = ane_matmul_bwd_compile(96, 1, S0);      CHK(bwd->head_bwd,"head_bwd");
    ane_matmul_bwd_rewire_w(bwd->head_bwd, net->head->w_surf);
    ane_matmul_bwd_rewire_x(bwd->head_bwd, net->head->x_surf);

    bwd->opt_stem  = ane_adam_compile(96*64,   lr,b1,b2,eps); CHK(bwd->opt_stem,"adam_stem");
    bwd->opt_down1 = ane_adam_compile(192*384, lr,b1,b2,eps); CHK(bwd->opt_down1,"adam_down1");
    bwd->opt_down2 = ane_adam_compile(384*768, lr,b1,b2,eps); CHK(bwd->opt_down2,"adam_down2");
    bwd->opt_fuse1 = ane_adam_compile(192*576, lr,b1,b2,eps); CHK(bwd->opt_fuse1,"adam_fuse1");
    bwd->opt_fuse2 = ane_adam_compile(96*288,  lr,b1,b2,eps); CHK(bwd->opt_fuse2,"adam_fuse2");
    bwd->opt_head  = ane_adam_compile(96,      lr,b1,b2,eps); CHK(bwd->opt_head,"adam_head");

    ane_adam_rewire_w(bwd->opt_stem,  net->stem->mm->w_surf);
    ane_adam_rewire_w(bwd->opt_down1, net->down1->mm->w_surf);
    ane_adam_rewire_w(bwd->opt_down2, net->down2->mm->w_surf);
    ane_adam_rewire_w(bwd->opt_fuse1, net->fuse1->mm->w_surf);
    ane_adam_rewire_w(bwd->opt_fuse2, net->fuse2->mm->w_surf);
    ane_adam_rewire_w(bwd->opt_head,  net->head->w_surf);

    bwd->opt_attn_Wq=(ANEAdam**)calloc(nA,8); bwd->opt_attn_Wk=(ANEAdam**)calloc(nA,8);
    bwd->opt_attn_Wv=(ANEAdam**)calloc(nA,8); bwd->opt_attn_Wo=(ANEAdam**)calloc(nA,8);
    for (int i=0;i<nA;i++) {
        bwd->opt_attn_Wq[i]=ane_adam_compile(384*384,lr,b1,b2,eps);
        bwd->opt_attn_Wk[i]=ane_adam_compile(384*384,lr,b1,b2,eps);
        bwd->opt_attn_Wv[i]=ane_adam_compile(384*384,lr,b1,b2,eps);
        bwd->opt_attn_Wo[i]=ane_adam_compile(384*384,lr,b1,b2,eps);
        CHK(bwd->opt_attn_Wq[i]&&bwd->opt_attn_Wk[i]&&bwd->opt_attn_Wv[i]&&bwd->opt_attn_Wo[i],"adam_attn");
        ane_adam_rewire_w(bwd->opt_attn_Wq[i], net->attn[i]->kq->ioInputs[0]);
        ane_adam_rewire_w(bwd->opt_attn_Wk[i], net->attn[i]->kk->ioInputs[0]);
        ane_adam_rewire_w(bwd->opt_attn_Wv[i], net->attn[i]->kv->ioInputs[0]);
        ane_adam_rewire_w(bwd->opt_attn_Wo[i], net->attn[i]->kout->ioInputs[0]);
    }

    // dW scratch
    bwd->dw_stem=malloc(96*64*2); bwd->dw_down1=malloc(192*384*2); bwd->dw_down2=malloc(384*768*2);
    bwd->dw_fuse1=malloc(192*576*2); bwd->dw_fuse2=malloc(96*288*2); bwd->dw_head=malloc(96*2);
    bwd->dw_attn_Wq=((_Float16**)calloc(nA,8)); bwd->dw_attn_Wk=((_Float16**)calloc(nA,8));
    bwd->dw_attn_Wv=((_Float16**)calloc(nA,8)); bwd->dw_attn_Wo=((_Float16**)calloc(nA,8));
    for (int i=0;i<nA;i++) {
        bwd->dw_attn_Wq[i]=malloc(384*384*2); bwd->dw_attn_Wk[i]=malloc(384*384*2);
        bwd->dw_attn_Wv[i]=malloc(384*384*2); bwd->dw_attn_Wo[i]=malloc(384*384*2);
    }

    // Gradient scratch
    bwd->dy_head=malloc(1*S0*2);    bwd->dx_head=malloc(96*S0*2);
    bwd->dx_up3=malloc(96*S1*2);    bwd->dy_stage5=malloc(96*S1*2); bwd->dx_stage5=malloc(96*S1*2);
    bwd->d_skip1=malloc(96*S1*2);   bwd->d_up2=malloc(192*S1*2);   bwd->dx_up2=malloc(192*S2*2);
    bwd->dy_stage4=malloc(192*S2*2);bwd->dx_stage4=malloc(192*S2*2);
    bwd->d_skip2=malloc(192*S2*2);  bwd->d_up1=malloc(384*S2*2);   bwd->dx_up1=malloc(384*S3*2);
    bwd->dy_attn=malloc(384*S3*2);  bwd->dx_attn=malloc(384*S3*2);
    bwd->dy_stage3=malloc(384*S3*2);bwd->dx_stage3=malloc(384*S3*2);
    bwd->dy_stage2=malloc(192*S2*2);bwd->dx_stage2=malloc(192*S2*2);
    bwd->dy_stage1=malloc(96*S1*2); bwd->dx_stage1=malloc(96*S1*2); bwd->dx_stem=malloc(3*S0*2);
    bwd->tmp2=malloc(192*S2*2);     bwd->tmp3=malloc(384*S3*2);

#undef CHK
    printf("[UNetLargeBwd] compiled OK (checkpointed=%d)\n", checkpointed);
    return bwd;
}

// ---- Save forward activations (skipped for checkpointed blocks) ----

static void ane_unet_large_bwd_save_fwd(ANEUNetLargeBwd *bwd, ANEUNetLarge *net, int H) {
#define SAVE_STAGE(arr_bwd, arr_fwd, nb, Hb) do { \
    for (int i=0;i<nb;i++) \
        if (!arr_bwd[i]->checkpointed) ane_convnext_save_fwd(arr_bwd[i], arr_fwd[i], Hb); \
} while(0)
    SAVE_STAGE(bwd->stage1_bwd, net->stage1, net->nB1, H/2);
    SAVE_STAGE(bwd->stage2_bwd, net->stage2, net->nB2, H/4);
    SAVE_STAGE(bwd->stage3_bwd, net->stage3, net->nB3, H/8);
    SAVE_STAGE(bwd->stage4_bwd, net->stage4, net->nB4, H/4);
    SAVE_STAGE(bwd->stage5_bwd, net->stage5, net->nB5, H/2);
#undef SAVE_STAGE
    for (int i=0;i<net->nA;i++) ane_attn_bwd_save_fwd(bwd->attn_bwd[i], net->attn[i]);
}

// ---- Backward pass ----

static void _large_run_stage_bwd(ANEConvNeXtBwd **bwds, ANEConvNeXt **fwds, int nB,
                                  const _Float16 *dy_last, _Float16 *dx_first,
                                  _Float16 *tmp, int H, int t) {
    const _Float16 *dy = dy_last;
    for (int i=nB-1; i>=0; i--) {
        _Float16 *dx = (i==0) ? dx_first : tmp;
        ane_convnext_bwd_eval(bwds[i], fwds[i], dy, dx, H, t);
        dy = dx;
    }
}

static void _large_adam_copy_wnew(ANEAdam *opt, IOSurfaceRef w_surf, int N) {
    IOSurfaceLock(opt->w_new_surf, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(w_surf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(w_surf), IOSurfaceGetBaseAddress(opt->w_new_surf), (size_t)N*2);
    IOSurfaceUnlock(w_surf, 0, NULL);
    IOSurfaceUnlock(opt->w_new_surf, kIOSurfaceLockReadOnly, NULL);
}

static float ane_unet_large_bwd_eval(ANEUNetLargeBwd *bwd, ANEUNetLarge *net,
                                      const _Float16 *p, const _Float16 *t_mask, int t) {
    int H=net->H, S0=net->S0, S1=net->S1, S2=net->S2, S3=net->S3;
    int nB1=net->nB1,nB2=net->nB2,nB3=net->nB3,nB4=net->nB4,nB5=net->nB5,nA=net->nA;

    float loss = l1bce_loss_and_grad(p, t_mask, bwd->dy_head, S0);

    // Head bwd
    ane_matmul_bwd_write_dy(bwd->head_bwd, bwd->dy_head);
    ane_matmul_bwd_eval(bwd->head_bwd);
    ane_matmul_bwd_read_dx(bwd->head_bwd, bwd->dx_head);
    ane_matmul_bwd_read_dw(bwd->head_bwd, bwd->dw_head);
    ane_adam_rewire_dw(bwd->opt_head, bwd->head_bwd->dw_surf);
    ane_adam_step(bwd->opt_head, t);
    _large_adam_copy_wnew(bwd->opt_head, net->head->w_surf, 96);

    ane_upsample2x_bwd(bwd->dx_head, bwd->dx_up3, 96, H/2);

    _large_run_stage_bwd(bwd->stage5_bwd, net->stage5, nB5,
                         bwd->dx_up3, bwd->dx_stage5, bwd->dy_stage5, H/2, t);

    ane_fuse_bwd_eval(bwd->fuse2_bwd, bwd->dx_stage5, bwd->d_skip1, bwd->d_up2);
    ane_fuse_bwd_read_dw(bwd->fuse2_bwd, bwd->dw_fuse2);
    ane_adam_rewire_dw(bwd->opt_fuse2, bwd->fuse2_bwd->mm_bwd->dw_surf);
    ane_adam_step(bwd->opt_fuse2, t);
    _large_adam_copy_wnew(bwd->opt_fuse2, net->fuse2->mm->w_surf, 96*288);

    ane_upsample2x_bwd(bwd->d_up2, bwd->dx_up2, 192, H/4);

    _large_run_stage_bwd(bwd->stage4_bwd, net->stage4, nB4,
                         bwd->dx_up2, bwd->dx_stage4, bwd->dy_stage4, H/4, t);

    ane_fuse_bwd_eval(bwd->fuse1_bwd, bwd->dx_stage4, bwd->d_skip2, bwd->d_up1);
    ane_fuse_bwd_read_dw(bwd->fuse1_bwd, bwd->dw_fuse1);
    ane_adam_rewire_dw(bwd->opt_fuse1, bwd->fuse1_bwd->mm_bwd->dw_surf);
    ane_adam_step(bwd->opt_fuse1, t);
    _large_adam_copy_wnew(bwd->opt_fuse1, net->fuse1->mm->w_surf, 192*576);

    ane_upsample2x_bwd(bwd->d_up1, bwd->dx_up1, 384, H/8);

    // Attn bwd
    memcpy(bwd->dy_attn, bwd->dx_up1, (size_t)384*S3*2);
    for (int i=nA-1; i>=0; i--) {
        ane_attn_bwd_eval(bwd->attn_bwd[i], net->attn[i],
                          bwd->dy_attn, bwd->dx_attn,
                          bwd->dw_attn_Wq[i], bwd->dw_attn_Wk[i],
                          bwd->dw_attn_Wv[i], bwd->dw_attn_Wo[i]);
        // Adam attn
        IOSurfaceLock(bwd->opt_attn_Wq[i]->dw_surf,0,NULL);
        memcpy(IOSurfaceGetBaseAddress(bwd->opt_attn_Wq[i]->dw_surf),bwd->dw_attn_Wq[i],384*384*2);
        IOSurfaceUnlock(bwd->opt_attn_Wq[i]->dw_surf,0,NULL);
        IOSurfaceLock(bwd->opt_attn_Wk[i]->dw_surf,0,NULL);
        memcpy(IOSurfaceGetBaseAddress(bwd->opt_attn_Wk[i]->dw_surf),bwd->dw_attn_Wk[i],384*384*2);
        IOSurfaceUnlock(bwd->opt_attn_Wk[i]->dw_surf,0,NULL);
        IOSurfaceLock(bwd->opt_attn_Wv[i]->dw_surf,0,NULL);
        memcpy(IOSurfaceGetBaseAddress(bwd->opt_attn_Wv[i]->dw_surf),bwd->dw_attn_Wv[i],384*384*2);
        IOSurfaceUnlock(bwd->opt_attn_Wv[i]->dw_surf,0,NULL);
        IOSurfaceLock(bwd->opt_attn_Wo[i]->dw_surf,0,NULL);
        memcpy(IOSurfaceGetBaseAddress(bwd->opt_attn_Wo[i]->dw_surf),bwd->dw_attn_Wo[i],384*384*2);
        IOSurfaceUnlock(bwd->opt_attn_Wo[i]->dw_surf,0,NULL);
        ane_adam_step(bwd->opt_attn_Wq[i],t); ane_adam_step(bwd->opt_attn_Wk[i],t);
        ane_adam_step(bwd->opt_attn_Wv[i],t); ane_adam_step(bwd->opt_attn_Wo[i],t);
        _large_adam_copy_wnew(bwd->opt_attn_Wq[i],net->attn[i]->kq->ioInputs[0],384*384);
        _large_adam_copy_wnew(bwd->opt_attn_Wk[i],net->attn[i]->kk->ioInputs[0],384*384);
        _large_adam_copy_wnew(bwd->opt_attn_Wv[i],net->attn[i]->kv->ioInputs[0],384*384);
        _large_adam_copy_wnew(bwd->opt_attn_Wo[i],net->attn[i]->kout->ioInputs[0],384*384);
        memcpy(bwd->dy_attn, bwd->dx_attn, (size_t)384*S3*2);
    }

    _large_run_stage_bwd(bwd->stage3_bwd, net->stage3, nB3,
                         bwd->dy_attn, bwd->dx_stage3, bwd->tmp3, H/8, t);

    // down2 bwd + adam
    ane_down_bwd_eval(bwd->down2_bwd, bwd->dx_stage3, bwd->dy_stage2);
    ane_down_bwd_read_dw(bwd->down2_bwd, bwd->dw_down2);
    ane_adam_rewire_dw(bwd->opt_down2, bwd->down2_bwd->mm_bwd->dw_surf);
    ane_adam_step(bwd->opt_down2, t);
    _large_adam_copy_wnew(bwd->opt_down2, net->down2->mm->w_surf, 384*768);

    _large_run_stage_bwd(bwd->stage2_bwd, net->stage2, nB2,
                         bwd->dy_stage2, bwd->dx_stage2, bwd->tmp2, H/4, t);

    // down1 bwd + adam
    ane_down_bwd_eval(bwd->down1_bwd, bwd->dx_stage2, bwd->dy_stage1);
    ane_down_bwd_read_dw(bwd->down1_bwd, bwd->dw_down1);
    ane_adam_rewire_dw(bwd->opt_down1, bwd->down1_bwd->mm_bwd->dw_surf);
    ane_adam_step(bwd->opt_down1, t);
    _large_adam_copy_wnew(bwd->opt_down1, net->down1->mm->w_surf, 192*384);

    _large_run_stage_bwd(bwd->stage1_bwd, net->stage1, nB1,
                         bwd->dy_stage1, bwd->dx_stage1, bwd->dy_stage1, H/2, t);

    // stem bwd + adam
    ane_stem_bwd_eval(bwd->stem_bwd, bwd->dx_stage1, bwd->dx_stem);
    ane_stem_bwd_read_dw(bwd->stem_bwd, bwd->dw_stem);
    ane_adam_rewire_dw(bwd->opt_stem, bwd->stem_bwd->mm_bwd->dw_surf);
    ane_adam_step(bwd->opt_stem, t);
    _large_adam_copy_wnew(bwd->opt_stem, net->stem->mm->w_surf, 96*64);

    return loss;
}
