# ane-train

Train neural networks on Apple Neural Engine.

## What

A framework for training CNNs and transformers entirely on ANE, leaving the GPU free. All ops are header-only C/ObjC — no Python, no PyTorch, no CoreML tools.

ANE only runs CoreML models. We generate [MIL](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html) text at runtime, compile via private API (`modelWithMILText:`), and execute with `evaluateWithModel:`. Weights live in IOSurfaces — compile once, train forever.

## Build & Run

```bash
make
DATA_ROOT=/path/to/train ./train_unet
```

Data layout: `$DATA_ROOT/{img,mask}/<sequence>/*.png` (RGB + grayscale mask pairs).

## Op Library

All ops are header-only with forward and backward:

| Op | ANE | CPU | Notes |
|----|-----|-----|-------|
| Matmul | x | | Core building block |
| LayerNorm | x | | 5-dispatch decomposition |
| Adam | x | | 3 kernels (m, v, w) |
| SiLU / GELU | x | | Element-wise |
| Sigmoid | x | | Element-wise |
| Add | x | | Residual connections |
| Self-attention | mixed | mixed | QK^T on CPU, AV on ANE |
| Depthwise conv | | x | NEON — ANE can't do grouped conv with runtime W |
| Upsample 2x | | x | NEON nearest-neighbor |
| Stem / Downsample | mixed | mixed | im2col on CPU, matmul on ANE |
| Loss (L1+BCE) | | x | With gradients |

## Requirements

- macOS 26+ (Apple Silicon)
- Xcode / Command Line Tools

See `ANE_TRAINING.md` for the full constraint cheatsheet.

## Credits

Built on top of [maderix/ane](https://github.com/maderix/ane) — the original ANE runtime reverse-engineering work that made all of this possible.
