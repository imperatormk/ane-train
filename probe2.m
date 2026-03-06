#import <Foundation/Foundation.h>
#include "ane_runtime.h"
#include "mil_gen.h"

int main(void) { @autoreleasepool {
    ane_init();
    const int N = 96;
    size_t sn = 2048;
    size_t ins4[4] = {sn,sn,sn,sn};
    ANEKernel *k_w = ane_compile([mil_gen_adam_w(N,1e-4f) dataUsingEncoding:NSUTF8StringEncoding],nil,4,ins4,1,&sn);
    void (^fill)(IOSurfaceRef,float) = ^(IOSurfaceRef s,float v){ _Float16 fv=(_Float16)v; IOSurfaceLock(s,0,NULL); uint8_t*b=(uint8_t*)IOSurfaceGetBaseAddress(s); for(size_t i=0;i+1<IOSurfaceGetAllocSize(s);i+=2)memcpy(b+i,&fv,2); IOSurfaceUnlock(s,0,NULL); };
    float (^rd)(IOSurfaceRef) = ^(IOSurfaceRef s){ _Float16 v; IOSurfaceLock(s,kIOSurfaceLockReadOnly,NULL); memcpy(&v,(uint8_t*)IOSurfaceGetBaseAddress(s)+3*2,2); IOSurfaceUnlock(s,kIOSurfaceLockReadOnly,NULL); return (float)v; };

    // ioInputs[0]=W confirmed. Test slots 1,2,3 as (mn,vn,lr).
    // W=10, m=1, v=4(sqrt=2), lr=3 → Wn = 10 - 3*1/2 = 8.5
    printf("Test W=10 v=4 m=1 lr=3 → expected Wn[3]=8.5\n");
    int perms[6][3] = {{1,2,3},{1,3,2},{2,1,3},{2,3,1},{3,1,2},{3,2,1}};
    const char *names[6][3] = {{"mn","vn","lr"},{"mn","lr","vn"},{"vn","mn","lr"},{"vn","lr","mn"},{"lr","mn","vn"},{"lr","vn","mn"}};
    for (int p=0;p<6;p++) {
        int mn_s=perms[p][0], vn_s=perms[p][1], lr_s=perms[p][2];
        fill(k_w->ioInputs[0],10.f);
        fill(k_w->ioInputs[mn_s],1.f);
        fill(k_w->ioInputs[vn_s],4.f);
        fill(k_w->ioInputs[lr_s],3.f);
        ane_eval(k_w);
        float wn=rd(k_w->ioOutputs[0]);
        printf("  slot1=%s slot2=%s slot3=%s → Wn[3]=%.4f %s\n",
               names[p][0],names[p][1],names[p][2],wn,fabsf(wn-8.5f)<0.1f?"*** MATCH ***":"");
    }
    return 0;
}}
