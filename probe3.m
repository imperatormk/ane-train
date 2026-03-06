#import <Foundation/Foundation.h>
#include "ane_runtime.h"
#include "mil_gen.h"

int main(void) { @autoreleasepool {
    ane_init();
    const int N = 96;
    size_t sn = 2048, ins4[4]={sn,sn,sn,sn};
    ANEKernel *k_w = ane_compile([mil_gen_adam_w(N,1e-4f) dataUsingEncoding:NSUTF8StringEncoding],nil,4,ins4,1,&sn);
    void (^fill)(IOSurfaceRef,float) = ^(IOSurfaceRef s,float v){ _Float16 fv=(_Float16)v; IOSurfaceLock(s,0,NULL); uint8_t*b=(uint8_t*)IOSurfaceGetBaseAddress(s); for(size_t i=0;i+1<IOSurfaceGetAllocSize(s);i+=2)memcpy(b+i,&fv,2); IOSurfaceUnlock(s,0,NULL); };
    float (^rd)(IOSurfaceRef) = ^(IOSurfaceRef s){ _Float16 v; IOSurfaceLock(s,kIOSurfaceLockReadOnly,NULL); memcpy(&v,(uint8_t*)IOSurfaceGetBaseAddress(s)+3*2,2); IOSurfaceUnlock(s,kIOSurfaceLockReadOnly,NULL); return (float)v; };

    // Candidates: [0]=W, [2]=lr. slot1=mn,slot3=vn OR slot1=vn,slot3=mn
    // Test: W=10, m=2, v=9(sqrt=3), lr=3 → Wn = 10 - 3*2/3 = 8.0
    printf("W=10 m=2 v=9 lr=3 → expected Wn=8.0\n");
    // Candidate A: slot1=mn, slot2=lr, slot3=vn
    fill(k_w->ioInputs[0],10.f); fill(k_w->ioInputs[1],2.f); fill(k_w->ioInputs[2],3.f); fill(k_w->ioInputs[3],9.f);
    ane_eval(k_w);
    printf("  A [0]=W [1]=mn [2]=lr [3]=vn → Wn=%.4f\n", rd(k_w->ioOutputs[0]));
    // Candidate B: slot1=vn, slot2=lr, slot3=mn
    fill(k_w->ioInputs[0],10.f); fill(k_w->ioInputs[1],9.f); fill(k_w->ioInputs[2],3.f); fill(k_w->ioInputs[3],2.f);
    ane_eval(k_w);
    printf("  B [0]=W [1]=vn [2]=lr [3]=mn → Wn=%.4f\n", rd(k_w->ioOutputs[0]));
    return 0;
}}
