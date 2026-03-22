# onnx-infer

PROJECT MIGRATED TO GITLAB: https://gitlab.com/micahcc/onnx-infer

This is almost purely because they have 10GB of free git lfs and I don't want to pay to host test models :/

Pure Rust ONNX inference engine targeting computer vision models. Zero heap
allocations after warmup — all tensor buffers, intermediate values, and output
maps are pre-allocated and reused across forward passes.

## Tested models

| Model            | Opset       | Variants                  |
| ---------------- | ----------- | ------------------------- |
| MNIST            | 1, 7, 8, 12 | float, int8               |
| MobileNet v2     | 7, 12       | float, int8, QDQ          |
| Tiny YOLOv2      | 7, 8        |                           |
| Tiny YOLOv3      | 11          | exercises Loop op         |
| YOLOv4           | 11          |                           |
| SSD MobileNet v1 | 12          |                           |
| Faster R-CNN     | 12          |                           |
| BiDAF            | 9           | NLP reading comprehension |

## Supported operators

Abs, Add, ArgMax, BatchNormalization, Cast, CategoryMapper, Ceil, Clip,
Compress, Concat, Constant, ConstantOfShape, Conv, DequantizeLinear, Div,
Dropout, Equal, Exp, Expand, Flatten, Floor, Gather, Gemm, GlobalAveragePool,
Greater, Hardmax, Identity, If, LeakyRelu, Less, Log, Loop, LSTM, MatMul, Max,
MaxPool, Min, Mul, NonMaxSuppression, NonZero, QLinearAdd, QLinearConv,
QLinearGlobalAveragePool, QLinearMatMul, QuantizeLinear, Range, ReduceMax,
ReduceMin, ReduceSum, Relu, Reshape, Resize, RoiAlign, Round, Scan,
ScatterElements, Shape, Sigmoid, Slice, Softmax, Split, Sqrt, Squeeze, Sub,
Sum, Tanh, Tile, TopK, Transpose, Unsqueeze, Where

## Library usage

```rust
use std::collections::HashMap;
use onnx_infer::{InferenceEngine, Tensor, dims};

let model_bytes = std::fs::read("model.onnx").unwrap();
let mut engine = InferenceEngine::new(&model_bytes).unwrap();

let input = Tensor::new(dims![1, 1, 28, 28], vec![0.0; 784]);
let mut inputs = HashMap::new();
inputs.insert("Input3".to_string(), input);

engine.run(inputs).unwrap();

let output = &engine.outputs["Plus214_Output_0"];
println!("dims: {:?}", output.dims);
println!("data: {:?}", &output.floats()[..10]);
```

## CLI

```
cargo run --features cli --bin onnx-infer -- model.onnx input_name=input.pb
cargo run --features cli --bin onnx-infer -- model.onnx input_name=image.png --grayscale --shape 1,1,28,28
```

Accepts `.pb` (serialized TensorProto) or image files (png, jpg, bmp, etc),
though image preprocessing is not yet fully implemented.

## Tests

Test fixtures are sourced from [onnx/models](https://github.com/onnx/models)
and stored in `fixtures/` using Git LFS. You'll need
[Git LFS](https://git-lfs.com/) installed to pull them:

```
git lfs install
git lfs pull
cargo test --release
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

BSD 3-Clause
