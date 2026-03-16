# onnx-infer

Pure Rust ONNX inference engine targeting computer vision models. Zero heap
allocations after warmup — all tensor buffers, intermediate values, and output
maps are pre-allocated and reused across forward passes.

## Supported operators

- Constant, Reshape, Flatten, Identity
- Conv (with auto_pad SAME_UPPER/SAME_LOWER, groups, dilations)
- Relu, LeakyRelu, Sigmoid, Exp, Ceil, Round, Clip
- MaxPool, GlobalAveragePool
- Add, Sub, Mul, Div (numpy-style broadcasting)
- MatMul, Gemm, Softmax
- BatchNormalization
- Concat, Gather, Slice, Squeeze, Unsqueeze, Transpose, Tile, Resize
- Shape, Cast
- Loop (with scan outputs and early termination)
- NonMaxSuppression
- DequantizeLinear, QuantizeLinear
- QLinearConv, QLinearMatMul, QLinearAdd, QLinearGlobalAveragePool

## Library usage

```rust
use std::collections::HashMap;
use onnx_infer::{InferenceEngine, Tensor};

let model_bytes = std::fs::read("model.onnx").unwrap();
let mut input_sizes = HashMap::new();
input_sizes.insert("Input3".to_string(), vec![1, 1, 28, 28]);
let mut engine = InferenceEngine::new(&model_bytes, input_sizes).unwrap();

let input = Tensor::new(vec![1, 1, 28, 28], vec![0.0; 784]);
let mut inputs = HashMap::new();
inputs.insert("Input3".to_string(), input);

engine.run(inputs).unwrap();

// Read results directly from the engine's output buffer (no copies)
let output = &engine.outputs["Plus214_Output_0"];
println!("dims: {:?}", output.dims);
println!("data: {:?}", &output.floats()[..10]);
```

## CLI

```
cargo run --bin onnx-infer -- model.onnx input_name=input.pb
cargo run --bin onnx-infer -- model.onnx input_name=image.png --grayscale --shape 1,1,28,28
```

Accepts `.pb` (serialized TensorProto) or image files (png, jpg, bmp, etc).

## Tests

The build script automatically downloads and caches test fixtures from
the [onnx/models](https://github.com/onnx/models) repository.

```
cargo test
```

Tested models: MNIST (opset 1/7/8/12, including int8), MobileNetV2 (opset
7/12, including int8 and QDQ), Tiny YOLOv2 (opset 7/8), Tiny YOLOv3 (opset
11, exercises Loop op).

## License

BSD 3-Clause
