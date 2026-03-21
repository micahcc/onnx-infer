# Changelog

## 2026-03-20 — Remove XNNPACK support

Removed the `xnnpack` feature and all associated code. XNNPACK subgraph
compilation had unresolved numerical accuracy issues across detection models
and layout-sensitive ops (Softmax axis, NHWC cascading errors). The feature
may return in the future once these are addressed.

Removed files: `xnnpack_subgraph.rs`, `xnnpack_ffi.rs`, `nix/xnnpack.nix`.
Removed `bindgen` build dependency. Cleaned up CI, nix flake, and CLAUDE.md.

## 2026-03-20 — CI for XNNPACK and BLAS, XNNPACK bug fixes, runtime disable flag

### CI
- Added separate CI jobs for no-features, BLAS (OpenBLAS), and XNNPACK
- All test jobs fetch Git LFS fixtures
- XNNPACK job uses `nix develop` to get pre-built XNNPACK
- Removed `--all-features` from clippy (was broken without native libs)

### XNNPACK bug fixes
- Fixed Gemm `XNN_FLAG_TRANSPOSE_WEIGHTS` logic (was inverted): ONNX `transB=1`
  means weight is `[O,I]` matching XNNPACK default, `transB=0` means `[I,O]`
  needing the transpose flag. This fixed all ResNet models producing all-zero
  output (48 XNNPACK tests pass, up from 46)
- Fixed compile errors: `vec_f32` closure return type, unused `mut`,
  moved `w_scales` value in QLinearConv

### Runtime flag
- `XNNPACK_DISABLE=1` env var disables XNNPACK acceleration at runtime
  (falls back to CPU for all ops), useful for testing and debugging

## 2026-03-20 — 41 new vision model fixtures, 6 new operators, Softmax opset fix

Added 41 vision models from ONNX Model Zoo (classification, object detection,
body analysis) as test fixtures with input/output validation.

### New operators
- LRN (Local Response Normalization)
- AveragePool
- ReduceMean
- PRelu (Parametric ReLU)
- Not (logical)
- And (logical)

### Bug fix: Softmax opset versioning
Softmax default axis changed between opsets: axis=1 for opset<13, axis=-1
for opset>=13. The engine was always using -1, causing incorrect results for
older models (e.g. SqueezeNet 1.0 opset 12). Fixed by threading opset version
from ModelProto through Graph to plan builder.

### Infrastructure
- `onnx_ir::Graph` now carries `opset_version`
- `convert_graph_with_opset()` accepts opset version from model proto
- `build_node_with_opset()` in plan.rs for opset-aware layer construction
- New test helpers: `run_fixture_argmax`, `run_multi_io_fixture_with_tol`
- `run_multi_io_fixture` handles mixed Float/Int64 dtype comparisons
- 56 tests pass, 3 ignored (MaskRCNN needs ConvTranspose, 2 deep segmentation
  models with accumulated FP divergence)

## 2026-03-19 18:30 — Deferred plan building & aggressive constant folding

Engine now defers plan building until first `run()` when input shapes are
incomplete. Uses actual input tensor values for aggressive constant folding,
eliminating entire shape-computation chains at build time.

- `InferenceEngine` stores the graph separately from the plan
- Plan rebuilds automatically when input shapes change
- `Plan::build_full` accepts `input_values` for build-time folding
- Loop ops fold when all inputs are known (arange grid generation)
- Int64 binary ops (Add, Sub, Mul, Div) now handled in constant folding
- tiny-yolov3-11: 96 nodes folded (was 30), 183 → 96 fewer plan ops
- All 18 tests pass

## 2026-03-19 17:45 — XNNPACK support for more op types

Added XNNPACK subgraph support for: Transpose, Identity, Unsqueeze, Squeeze,
Slice, Resize, Round, Cast (float→float), ReduceMin.

- Transpose → `xnn_define_static_transpose`
- Identity → `xnn_define_copy`
- Unsqueeze/Squeeze → `xnn_define_static_reshape`
- Slice → `xnn_define_static_slice`
- Resize → `xnn_define_static_resize_bilinear_2d`
- Round → `xnn_define_bankers_rounding`
- Cast (float) → `xnn_define_convert`
- ReduceMin → `xnn_define_static_reduce` with `xnn_reduce_min`

Also expanded constant-folding to cover: Transpose, Neg, Mul, Div, Add, Sub,
Floor, Ceil, Round, Tile, Exp, ReduceMin/Max/Sum, Equal, Less, Greater, Where,
Range, Expand, Min, Max.

## 2026-03-19 16:45 — XNNPACK quantized op support

Added XNNPACK subgraph support for quantized (int8) ONNX operators:
QuantizeLinear, DequantizeLinear, QLinearConv, QLinearMatMul, QLinearAdd.

- Uses XNNPACK's native quantized tensor API (`xnn_define_quantized_tensor_value`,
  `xnn_define_channelwise_quantized_tensor_value`) with automatic ONNX uint8→qint8
  conversion
- Same XNNPACK operators (conv2d, fully_connected, binary) auto-dispatch to
  quantized kernels based on tensor datatype
- `xnn_define_convert` handles QuantizeLinear (fp32→qint8) and DequantizeLinear
  (qint8→fp32)
- Quantization info propagates through layout-preserving ops (MaxPool, Reshape,
  Flatten, GlobalAvgPool, unary ops)
- mnist-12-int8 now 100% XNNPACK-eligible (was 27%)

## 2026-03-19 15:30 — Constant-folding eliminates plan nodes

Ops whose outputs are fully determined at plan-build time (Constant, Shape,
Gather on constants, etc.) are no longer added to the execution plan. Their
computed values go directly into initializers, available at runtime without
redundant execution. This removes the nodes that previously broke XNNPACK
subgraph runs.

Key impact:
- mobilenetv2-12: 100% XNNPACK-eligible (Constant nodes for Clip min/max no longer break runs)
- mnist-1: 12 constant nodes folded away
- yolov4-11: 28 constant nodes folded, now 98% eligible
- CPU-path benchmark: ~1-4% speedup (mobilenetv2-12 best at -4.1%)
