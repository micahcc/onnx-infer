# onnx-infer

Pure Rust ONNX inference engine targeting computer vision models.

## Supported operators

- Constant, Reshape, Flatten
- Conv (with auto_pad SAME_UPPER/SAME_LOWER, groups, dilations)
- Relu, MaxPool, AveragePool / GlobalAveragePool
- Add, Sub, Mul, Div (numpy-style broadcasting + legacy ONNX broadcast/axis)
- MatMul, Softmax

## Library usage

```rust
use std::collections::HashMap;
use onnx_infer::{InferenceEngine, Tensor};

let model_bytes = std::fs::read("model.onnx").unwrap();
let engine = InferenceEngine::from_bytes(&model_bytes).unwrap();

let input = Tensor::new(vec![1, 1, 28, 28], vec![0.0; 784]);
let mut inputs = HashMap::new();
inputs.insert("Input3".to_string(), input);

let outputs = engine.run(inputs).unwrap();
```

## CLI

```
cargo run --bin onnx-infer -- model.onnx input_name=input.pb
cargo run --bin onnx-infer -- model.onnx input_name=image.png --grayscale --shape 1,1,28,28
```

Accepts `.pb` (serialized TensorProto) or image files (png, jpg, bmp, etc).

## Tests

The build script automatically downloads and caches MNIST test fixtures from
the [onnx/models](https://github.com/onnx/models) repository.

```
cargo test
```

## License

BSD 3-Clause
