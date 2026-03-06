// data_utils.h — Image loading + dataset utilities (macOS AppKit)
//
// Requires: -framework AppKit
//
#pragma once
#import <AppKit/AppKit.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- Bilinear resize ----
// src: [sc, sh, sw] uint8, dst: [sc, dh, dw] float32 (normalized 0..1)
static void bilinear_resize(const uint8_t *src, int sc, int sh, int sw,
                             float *dst, int dh, int dw) {
    float sy = (float)sh/dh, sx = (float)sw/dw;
    for (int c = 0; c < sc; c++)
    for (int y = 0; y < dh; y++)
    for (int x = 0; x < dw; x++) {
        float fy = (y+0.5f)*sy-0.5f, fx = (x+0.5f)*sx-0.5f;
        int y0 = (int)fy; float wy = fy-y0; if (y0 < 0) { y0=0; wy=0; }
        int x0 = (int)fx; float wx = fx-x0; if (x0 < 0) { x0=0; wx=0; }
        int y1 = y0+1 < sh ? y0+1 : sh-1;
        int x1 = x0+1 < sw ? x0+1 : sw-1;
        float v00 = src[c*sh*sw+y0*sw+x0]/255.f, v01 = src[c*sh*sw+y0*sw+x1]/255.f;
        float v10 = src[c*sh*sw+y1*sw+x0]/255.f, v11 = src[c*sh*sw+y1*sw+x1]/255.f;
        dst[c*dh*dw+y*dw+x] = (1-wy)*((1-wx)*v00+wx*v01) + wy*((1-wx)*v10+wx*v11);
    }
}

// ---- PNG loading ----
// Loads PNG at path, resizes to [nc, dh, dw] float32. Returns 1 on success.
static int load_png(const char *path, int nc, int dh, int dw, float *out) {
    NSString *p = [NSString stringWithUTF8String:path];
    NSBitmapImageRep *rep = (NSBitmapImageRep *)[NSBitmapImageRep imageRepWithContentsOfFile:p];
    if (!rep) { fprintf(stderr, "load_png: failed to load %s\n", path); return 0; }
    NSBitmapImageRep *conv = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL pixelsWide:rep.pixelsWide pixelsHigh:rep.pixelsHigh
        bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES isPlanar:NO
        colorSpaceName:NSCalibratedRGBColorSpace bytesPerRow:0 bitsPerPixel:0];
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithBitmapImageRep:conv]];
    [rep drawInRect:NSMakeRect(0, 0, rep.pixelsWide, rep.pixelsHigh)];
    [NSGraphicsContext restoreGraphicsState];
    int sh = (int)conv.pixelsHigh, sw = (int)conv.pixelsWide;
    uint8_t *tmp = malloc(nc * sh * sw);
    uint8_t *bmp = conv.bitmapData;
    for (int y = 0; y < sh; y++)
    for (int x = 0; x < sw; x++)
    for (int c = 0; c < nc; c++)
        tmp[c*sh*sw + y*sw + x] = bmp[(y*sw+x)*4 + c];
    bilinear_resize(tmp, nc, sh, sw, out, dh, dw);
    free(tmp);
    return 1;
}

// ---- Dataset pair ----
typedef struct { char img[512]; char mask[512]; } ImgPair;

// Scan data_root/{imgDir,maskDir}/<seq>/*.png and collect matched pairs.
// imgDir/maskDir default to "img"/"mask", override with DATA_IMG_DIR/DATA_MASK_DIR env vars.
static ImgPair *collect_pairs(const char *root, int *count) {
    ImgPair *pairs = malloc(100000 * sizeof(ImgPair));
    *count = 0;
    const char *id = getenv("DATA_IMG_DIR");  if (!id) id = "img";
    const char *md = getenv("DATA_MASK_DIR"); if (!md) md = "mask";
    NSString *imgRoot = [NSString stringWithFormat:@"%s/%s", root, id];
    NSString *maskRoot = [NSString stringWithFormat:@"%s/%s", root, md];
    NSArray *seqs = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:imgRoot error:nil];
    for (NSString *seq in seqs) {
        NSString *fd = [imgRoot stringByAppendingPathComponent:seq];
        NSArray *frames = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:fd error:nil];
        for (NSString *f in frames) {
            if (![f hasSuffix:@".png"]) continue;
            snprintf(pairs[*count].img, 512, "%s/%s/%s",
                     imgRoot.UTF8String, seq.UTF8String, f.UTF8String);
            snprintf(pairs[*count].mask, 512, "%s/%s/%s",
                     maskRoot.UTF8String, seq.UTF8String, f.UTF8String);
            (*count)++;
        }
    }
    return pairs;
}

// ---- fp32 → fp16 ----
static void f32_to_f16(const float *src, _Float16 *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = (_Float16)src[i];
}
