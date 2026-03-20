# CLAUDE.md

## Project Overview

Pure Rust ONNX inference engine for computer vision models. Zero heap allocations after warmup.

## Project Structure

```
src/
  lib.rs                    - Public API, tests for all fixture models
  inference_engine.rs       - Top-level engine: loads model, runs plan, manages values
  tensor_data.rs            - Tensor type (Dims = SmallVec<[usize; 8]>), f32/i64/string storage
  onnx_ir.rs                - ONNX protobuf → internal IR (Graph, Node, Attrs)
  layers/
    mod.rs                  - Layer trait, binary_op helper, quantize/dequantize helpers
    plan.rs                 - Builds execution plan from IR graph, constant-folding, XNNPACK compilation
    op_type.rs              - OpType enum, dtype/shape inference
    xnnpack_subgraph.rs     - XNNPACK subgraph compilation and execution (feature-gated)
    *.rs                    - One file per ONNX operator (conv.rs, matmul.rs, etc.)
  blas.rs                   - BLAS/Accelerate bindings for GEMM
  bin/                      - CLI tools (infer, dump_onnx, inspect_model, load_onnx)
proto/                      - ONNX protobuf definitions
fixtures/                   - Test models from onnx/models (Git LFS)
benches/                    - Criterion benchmarks
build.rs                    - Protobuf codegen, XNNPACK bindgen (when feature enabled)
```

## Key Patterns

- **Layer trait**: Each op implements `Layer::execute(&mut self, values, output)`
- **Precomputation**: Layers cache shape-dependent data, recompute if input shape changes
- **Constant folding**: `try_propagate_value()` in plan.rs folds Shape/Constant/Gather/etc. at build time; folded values go directly into initializers (skipping the plan)
- **XNNPACK**: Feature-gated (`xnnpack`). Subgraphs of consecutive compatible ops compiled into XNNPACK runtime. Handles NCHW↔NHWC internally. Supports float and quantized (qint8) ops. Requires `XNNPACK` env var pointing to built XNNPACK directory.

## Build & Test

```bash
cargo test --release              # Run all tests (needs Git LFS fixtures)
cargo test test_mnist12_set_0     # Run a single test
XNNPACK=/path/to/xnnpack cargo test --features xnnpack  # With XNNPACK
```

## Workflow Rules

- Every change must be documented in CHANGELOG.md (reverse chronological, with dates)
