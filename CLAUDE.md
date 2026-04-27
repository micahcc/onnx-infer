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
    plan.rs                 - Builds execution plan from IR graph, constant-folding
    op_type.rs              - OpType enum, dtype/shape inference
    *.rs                    - One file per ONNX operator (conv.rs, matmul.rs, etc.)
  blas.rs                   - BLAS/Accelerate bindings for GEMM
  bin/                      - CLI tools (infer, dump_onnx, inspect_model, load_onnx)
proto/                      - ONNX protobuf definitions
fixtures/                   - Test models from onnx/models (Git LFS)
benches/                    - Criterion benchmarks
build.rs                    - Protobuf codegen
```

## Key Patterns

- **Layer trait**: Each op implements `Layer::execute(&mut self, values, output)`
- **Precomputation**: Layers cache shape-dependent data, recompute if input shape changes
- **Constant folding**: `try_propagate_value()` in plan.rs folds Shape/Constant/Gather/etc. at build time; folded values go directly into initializers (skipping the plan)

## Build & Test

```bash
RUST_LOG=debug cargo test -- --nocapture > /tmp/onnx-infer-tests.log 2>&1
cargo test test_mnist12_set_0     # Run a single test
```

- Always write output to a temp file so nothing gets lost in scrollback.
- Slowest tests: resnet101_duc (~9min), fcn_resnet101 (~4min), ssd_12 (~3min).

## Debugging Guide

### Constant Folding & Shape Ops

`try_propagate_value()` in `plan.rs` folds ops into constants at plan time. **Shape ops must only be folded when their input is a known constant** (initializer or previously folded value). Shape ops on runtime tensors must execute at runtime because `shape_map` inference can be inaccurate — especially for models with dynamic spatial dimensions (Faster RCNN, Mask RCNN). A past bug folded Shape ops using `shape_map`, producing wrong reshape targets and causing Gather OOB panics.

### Diagnosing Shape/Index Panics

When you see panics like "range start index X out of range for slice of length Y" in Gather, Reshape, etc.:

1. The root cause is usually an **upstream tensor with wrong shape**, not the panicking op.
2. Add temporary debug prints in `execute_plan()` (`inference_engine.rs` ~line 293) to log tensor dims:
   ```rust
   eprintln!("DEBUG tensor {}: dims={:?}", output, out.dims.as_slice());
   ```
3. Use python to trace the ONNX graph and find which node produces a tensor:
   ```python
   import onnx
   model = onnx.load('path/to/model.onnx')
   for node in model.graph.node:
       if 'TENSOR_NAME' in node.output:
           print(f'{node.op_type} inputs={list(node.input)} outputs={list(node.output)}')
   ```
4. Compare runtime tensor shapes against expected. A mismatch points to a buggy op or incorrect constant folding.

### Common Pitfalls

- **Dynamic dimensions**: Detection models (RCNN family, YOLO with NMS) have many dynamic shapes. Be cautious with plan-time optimizations on these.
- **TopK + Gather patterns**: In detection models, TopK selects indices from scores, then Gather uses those indices on boxes. Both tensors must have matching sizes on the indexed axis. If they don't, look upstream for shape mismatches.
- **shape_map vs reality**: `shape_map` is populated by inference during plan building. It can be wrong when ops have complex shape logic (padding, ceil_mode, dynamic slicing). Only trust shapes of tensors that are actually materialized.

## Workflow Rules

- Every change must be documented in CHANGELOG.md (reverse chronological, with dates)
