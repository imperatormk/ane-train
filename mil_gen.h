// mil_gen.h — MIL text generators for ANE kernels
//
// Generates MIL program strings for: matmul (fwd/dx/dW), Adam (m/v/w), add.
// All use runtime inputs (IOSurface) — no baked weights, no recompile on update.
//
// MIL tensor layout: [N, C, S] for matmul, [1, N] for Adam.
// All fp16 (when g_fp16_io=1).
#pragma once
#import <Foundation/Foundation.h>
#include <math.h>

// ANE is fp16-only on all generations (M1–M4). fp32 falls back to GPU.
// g_fp16_io kept for compatibility with callers that check it.
extern int g_fp16_io;

// ============================================================================
// Matmul kernels
// ============================================================================

// Forward: y = W @ x
// W [1, C_out, C_in], x [1, C_in, S] → y [1, C_out, S]
static NSString *mil_gen_matmul_fwd(int C_in, int C_out, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(\n"
        "        tensor<fp16, [1, %d, %d]> W,\n"
        "        tensor<fp16, [1, %d, %d]> x) {\n"
        "        tensor<bool, []> ff = const()[name=tensor<string,[]>(\"ff\"), val=tensor<bool,[]>(false)];\n"
        "        tensor<fp16, [1, %d, %d]> y = matmul(transpose_x=ff, transpose_y=ff, x=W, y=x)"
            "[name=tensor<string,[]>(\"y\")];\n"
        "    } -> (y);\n"
        "}\n",
        C_out, C_in,
        C_in,  S,
        C_out, S];
}

// Backward dx: dx = W^T @ dy
// W [1, C_out, C_in], dy [1, C_out, S] → dx [1, C_in, S]
static NSString *mil_gen_matmul_dx(int C_in, int C_out, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(\n"
        "        tensor<fp16, [1, %d, %d]> W,\n"
        "        tensor<fp16, [1, %d, %d]> dy) {\n"
        "        tensor<bool, []> tt = const()[name=tensor<string,[]>(\"tt\"), val=tensor<bool,[]>(true)];\n"
        "        tensor<bool, []> ff = const()[name=tensor<string,[]>(\"ff\"), val=tensor<bool,[]>(false)];\n"
        "        tensor<fp16, [1, %d, %d]> dx = matmul(transpose_x=tt, transpose_y=ff, x=W, y=dy)"
            "[name=tensor<string,[]>(\"dx\")];\n"
        "    } -> (dx);\n"
        "}\n",
        C_out, C_in,
        C_out, S,
        C_in,  S];
}

// Weight gradient: dW = dy @ x^T
// Inputs:  dy [1, C_out, S], x [1, C_in, S]
// Output:  dW [1, C_out, C_in]
// ANE slot rule: in[0] ≤ in[1] → dy first (Co*S ≤ Ci*S when Co ≤ Ci)
static NSString *mil_gen_dW(int C_in, int C_out, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, %d]> dy, tensor<fp16, [1, %d, %d]> x) {\n"
        "        tensor<bool, []> tx = const()[name=tensor<string, []>(\"tx\"), val=tensor<bool, []>(false)];\n"
        "        tensor<bool, []> ty = const()[name=tensor<string, []>(\"ty\"), val=tensor<bool, []>(true)];\n"
        "        tensor<fp16, [1, %d, %d]> dW = matmul(transpose_x=tx, transpose_y=ty, x=dy, y=x)"
            "[name=tensor<string, []>(\"dW\")];\n"
        "    } -> (dW);\n"
        "}\n",
        C_out, S, C_in, S, C_out, C_in];
}

// ============================================================================
// Adam kernels (3 separate to avoid 4-input scalar-broadcast bug)
// ============================================================================

// Kernel 1: m_new = beta1*m + (1-beta1)*dW
static NSString *mil_gen_adam_m(int N, float beta1) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d]> dW, tensor<fp16, [1, %d]> m) {\n"
        "        tensor<fp16, []> b1  = const()[name=tensor<string,[]>(\"b1\"),  val=tensor<fp16,[]>(%.6f)];\n"
        "        tensor<fp16, []> b1c = const()[name=tensor<string,[]>(\"b1c\"), val=tensor<fp16,[]>(%.6f)];\n"
        "        tensor<fp16, [1, %d]> b1m  = mul(x=b1,  y=m)[name=tensor<string,[]>(\"b1m\")];\n"
        "        tensor<fp16, [1, %d]> b1dW = mul(x=b1c, y=dW)[name=tensor<string,[]>(\"b1dW\")];\n"
        "        tensor<fp16, [1, %d]> mn   = add(x=b1m, y=b1dW)[name=tensor<string,[]>(\"mn\")];\n"
        "    } -> (mn);\n"
        "}\n",
        N, N, beta1, 1.0f-beta1, N, N, N];
}

// Kernel 2: v_new = beta2*v + (1-beta2)*dW*dW
static NSString *mil_gen_adam_v(int N, float beta2) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d]> dW, tensor<fp16, [1, %d]> v) {\n"
        "        tensor<fp16, []> b2  = const()[name=tensor<string,[]>(\"b2\"),  val=tensor<fp16,[]>(%.6f)];\n"
        "        tensor<fp16, []> b2c = const()[name=tensor<string,[]>(\"b2c\"), val=tensor<fp16,[]>(%.6f)];\n"
        "        tensor<fp16, [1, %d]> dW2   = mul(x=dW,  y=dW)[name=tensor<string,[]>(\"dW2\")];\n"
        "        tensor<fp16, [1, %d]> b2v   = mul(x=b2,  y=v)[name=tensor<string,[]>(\"b2v\")];\n"
        "        tensor<fp16, [1, %d]> b2dW2 = mul(x=b2c, y=dW2)[name=tensor<string,[]>(\"b2dW2\")];\n"
        "        tensor<fp16, [1, %d]> vn    = add(x=b2v, y=b2dW2)[name=tensor<string,[]>(\"vn\")];\n"
        "    } -> (vn);\n"
        "}\n",
        N, N, beta2, 1.0f-beta2, N, N, N, N];
}

// Kernel 3: W_new = W*(1-wd) - lr * m_new / sqrt(v_new + eps)  (AdamW)
// eps >= 1e-4 required for fp16 (1e-8 underflows)
// wd (weight decay) and eps are baked constants. lr_t is a runtime input [1,N].
static NSString *mil_gen_adam_w(int N, float eps, float wd) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d]> W, tensor<fp16, [1, %d]> mn, tensor<fp16, [1, %d]> vn, tensor<fp16, [1, %d]> lr_in) {\n"
        "        tensor<fp16, []> owd   = const()[name=tensor<string,[]>(\"owd\"),  val=tensor<fp16,[]>(%.6f)];\n"
        "        tensor<fp16, []> eps_t = const()[name=tensor<string,[]>(\"eps\"),  val=tensor<fp16,[]>(%.6f)];\n"
        "        tensor<fp16, [1, %d]> Wd   = mul(x=W,    y=owd)[name=tensor<string,[]>(\"Wd\")];\n"
        "        tensor<fp16, [1, %d]> ve   = add(x=vn,   y=eps_t)[name=tensor<string,[]>(\"ve\")];\n"
        "        tensor<fp16, [1, %d]> sv   = sqrt(x=ve)[name=tensor<string,[]>(\"sv\")];\n"
        "        tensor<fp16, [1, %d]> step = real_div(x=mn, y=sv)[name=tensor<string,[]>(\"step\")];\n"
        "        tensor<fp16, [1, %d]> ls   = mul(x=lr_in, y=step)[name=tensor<string,[]>(\"ls\")];\n"
        "        tensor<fp16, [1, %d]> Wn   = sub(x=Wd,   y=ls)[name=tensor<string,[]>(\"Wn\")];\n"
        "    } -> (Wn);\n"
        "}\n",
        N, N, N, N, 1.0f - wd, eps, N, N, N, N, N, N];
}

// ============================================================================
// Element-wise add (residual connection)
// ============================================================================

// a[1,D,S] + b[1,D,S] → c[1,D,S]
static NSString *mil_gen_add(int D, int S) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, %d]> a, tensor<fp16, [1, %d, %d]> b) {\n"
        "        tensor<fp16, [1, %d, %d]> c = add(x=a, y=b)[name=tensor<string,[]>(\"c\")];\n"
        "    } -> (c);\n"
        "}\n",
        D, S, D, S, D, S];
}
