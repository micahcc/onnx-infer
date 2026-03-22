use std::collections::HashMap;
use std::fs;
use std::path::Path;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use onnx_infer::InferenceEngine;
use onnx_infer::Tensor;
use onnx_infer::onnx::TensorProto;
use prost::Message;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn load_single_input(base: &Path, model_file: &str) -> (InferenceEngine, HashMap<String, Tensor>) {
    load_single_input_opt(base, model_file, false)
}

#[derive(Clone, Copy, PartialEq)]
enum EngineMode {
    Plain,
    GraphOpt,
    #[cfg(feature = "xnnpack")]
    Xnnpack,
}

fn load_single_input_mode(
    base: &Path,
    model_file: &str,
    mode: EngineMode,
) -> (InferenceEngine, HashMap<String, Tensor>) {
    let model_bytes = fs::read(base.join(model_file)).expect("read model");

    let model = onnx_infer::onnx::ModelProto::decode(&model_bytes[..]).unwrap();
    let graph = model.graph.as_ref().unwrap();

    let test_dir = base.join("test_data_set_0");
    let pb_bytes = fs::read(test_dir.join("input_0.pb")).expect("read input");
    let proto = TensorProto::decode(&pb_bytes[..]).expect("decode tensor proto");
    let name = if proto.name.is_empty() {
        graph.input[0].name.clone()
    } else {
        proto.name.clone()
    };
    let input = Tensor::from_proto(&proto).expect("parse input");

    let engine = match mode {
        EngineMode::Plain => InferenceEngine::new(&model_bytes).expect("load model"),
        EngineMode::GraphOpt => InferenceEngine::with_graph_opt(&model_bytes).expect("load model"),
        #[cfg(feature = "xnnpack")]
        EngineMode::Xnnpack => InferenceEngine::with_xnnpack(&model_bytes).expect("load model"),
    };

    let mut inputs = HashMap::new();
    inputs.insert(name, input);
    (engine, inputs)
}

fn load_single_input_opt(
    base: &Path,
    model_file: &str,
    graph_opt: bool,
) -> (InferenceEngine, HashMap<String, Tensor>) {
    load_single_input_mode(base, model_file, if graph_opt { EngineMode::GraphOpt } else { EngineMode::Plain })
}

fn load_multi_input(base: &Path, model_file: &str) -> (InferenceEngine, HashMap<String, Tensor>) {
    load_multi_input_opt(base, model_file, false)
}

fn load_multi_input_mode(
    base: &Path,
    model_file: &str,
    mode: EngineMode,
) -> (InferenceEngine, HashMap<String, Tensor>) {
    let model_bytes = fs::read(base.join(model_file)).expect("read model");

    let model = onnx_infer::onnx::ModelProto::decode(&model_bytes[..]).unwrap();
    let graph = model.graph.as_ref().unwrap();

    let test_dir = base.join("test_data_set_0");
    let mut inputs = HashMap::new();
    for i in 0..graph.input.len() {
        let pb_path = test_dir.join(format!("input_{i}.pb"));
        if pb_path.exists() {
            let pb_bytes = fs::read(&pb_path).expect("read input");
            let proto = TensorProto::decode(&pb_bytes[..]).expect("decode tensor proto");
            let name = if proto.name.is_empty() {
                graph.input[i].name.clone()
            } else {
                proto.name.clone()
            };
            let input = Tensor::from_proto(&proto).expect("parse input");
            inputs.insert(name, input);
        }
    }

    let engine = match mode {
        EngineMode::Plain => InferenceEngine::new(&model_bytes).expect("load model"),
        EngineMode::GraphOpt => InferenceEngine::with_graph_opt(&model_bytes).expect("load model"),
        #[cfg(feature = "xnnpack")]
        EngineMode::Xnnpack => InferenceEngine::with_xnnpack(&model_bytes).expect("load model"),
    };
    (engine, inputs)
}

fn load_multi_input_opt(
    base: &Path,
    model_file: &str,
    graph_opt: bool,
) -> (InferenceEngine, HashMap<String, Tensor>) {
    load_multi_input_mode(base, model_file, if graph_opt { EngineMode::GraphOpt } else { EngineMode::Plain })
}

/// Run inference once and validate outputs match the expected fixture data.
/// Panics if any output diverges beyond tolerance.
fn validate_once(engine: &mut InferenceEngine, inputs: &HashMap<String, Tensor>, base: &Path, model_file: &str, test_set: usize, tol: f32) {
    let model_bytes = fs::read(base.join(model_file)).expect("read model");
    let model = onnx_infer::onnx::ModelProto::decode(&model_bytes[..]).unwrap();
    let graph = model.graph.as_ref().unwrap();

    engine.run(inputs.clone()).expect("inference");

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    for i in 0..graph.output.len() {
        let pb_path = test_dir.join(format!("output_{i}.pb"));
        if !pb_path.exists() {
            continue;
        }
        let expected = Tensor::from_proto_bytes(&fs::read(&pb_path).expect("read output"))
            .expect("parse output");
        let name = &graph.output[i].name;
        let output = engine
            .outputs
            .get(name)
            .unwrap_or_else(|| panic!("missing output {name}"));
        assert_eq!(output.dims, expected.dims, "shape mismatch for {name}");

        if output.dtype() == onnx_infer::DType::Float && expected.dtype() == onnx_infer::DType::Float {
            let got = output.floats().expect("float");
            let want = expected.floats().expect("float");
            let mut max_err: f32 = 0.0;
            let mut max_idx = 0;
            for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                    max_idx = j;
                }
            }
            assert!(
                max_err < tol,
                "output {name}[{max_idx}]: max_err={max_err}, got={}, want={}, tol={tol}",
                got[max_idx], want[max_idx]
            );
        }
    }
}

fn bench_mnist1(c: &mut Criterion) {
    let base = fixture("mnist-1");
    let (mut engine, inputs) = load_single_input(&base, "model.onnx");
    validate_once(&mut engine, &inputs, &base, "model.onnx", 0, 1e-4);
    c.bench_function("mnist-1", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mnist7(c: &mut Criterion) {
    let base = fixture("mnist-7");
    let (mut engine, inputs) = load_single_input(&base, "model.onnx");
    validate_once(&mut engine, &inputs, &base, "model.onnx", 0, 1e-4);
    c.bench_function("mnist-7", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mnist8(c: &mut Criterion) {
    let base = fixture("mnist-8");
    let (mut engine, inputs) = load_single_input(&base, "model.onnx");
    validate_once(&mut engine, &inputs, &base, "model.onnx", 0, 1e-4);
    c.bench_function("mnist-8", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mnist12(c: &mut Criterion) {
    let base = fixture("mnist-12");
    let (mut engine, inputs) = load_single_input(&base, "mnist-12.onnx");
    validate_once(&mut engine, &inputs, &base, "mnist-12.onnx", 0, 1e-4);
    c.bench_function("mnist-12", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mobilenetv2_7(c: &mut Criterion) {
    let base = fixture("mobilenetv2-7");
    let (mut engine, inputs) = load_single_input(&base, "mobilenetv2-7.onnx");
    validate_once(&mut engine, &inputs, &base, "mobilenetv2-7.onnx", 0, 5e-3);
    c.bench_function("mobilenetv2-7", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mobilenetv2_12(c: &mut Criterion) {
    let base = fixture("mobilenetv2-12");
    let (mut engine, inputs) = load_single_input(&base, "mobilenetv2-12.onnx");
    validate_once(&mut engine, &inputs, &base, "mobilenetv2-12.onnx", 0, 5e-3);
    c.bench_function("mobilenetv2-12", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_tinyyolov2_7(c: &mut Criterion) {
    let base = fixture("tinyyolov2-7");
    let (mut engine, inputs) = load_single_input(&base, "model.onnx");
    validate_once(&mut engine, &inputs, &base, "model.onnx", 0, 5e-3);
    c.bench_function("tinyyolov2-7", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_tinyyolov3_11(c: &mut Criterion) {
    let base = fixture("tiny-yolov3-11");
    let (mut engine, inputs) = load_multi_input(&base, "yolov3-tiny.onnx");
    validate_once(&mut engine, &inputs, &base, "yolov3-tiny.onnx", 0, 5e-3);
    c.bench_function("tiny-yolov3-11", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_yolov4_11(c: &mut Criterion) {
    let base = fixture("yolov4-11");
    let (mut engine, inputs) = load_single_input(&base, "yolov4.onnx");
    validate_once(&mut engine, &inputs, &base, "yolov4.onnx", 0, 5e-3);
    c.bench_function("yolov4-11", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

// --- graph-opt variants (BN fold + dead node removal) ---

fn bench_mnist12_graphopt(c: &mut Criterion) {
    let base = fixture("mnist-12");
    let (mut engine, inputs) = load_single_input_opt(&base, "mnist-12.onnx", true);
    validate_once(&mut engine, &inputs, &base, "mnist-12.onnx", 0, 1e-4);
    c.bench_function("mnist-12-graphopt", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mobilenetv2_7_graphopt(c: &mut Criterion) {
    let base = fixture("mobilenetv2-7");
    let (mut engine, inputs) = load_single_input_opt(&base, "mobilenetv2-7.onnx", true);
    validate_once(&mut engine, &inputs, &base, "mobilenetv2-7.onnx", 0, 5e-3);
    c.bench_function("mobilenetv2-7-graphopt", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mobilenetv2_12_graphopt(c: &mut Criterion) {
    let base = fixture("mobilenetv2-12");
    let (mut engine, inputs) = load_single_input_opt(&base, "mobilenetv2-12.onnx", true);
    validate_once(&mut engine, &inputs, &base, "mobilenetv2-12.onnx", 0, 5e-3);
    c.bench_function("mobilenetv2-12-graphopt", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_tinyyolov2_7_graphopt(c: &mut Criterion) {
    let base = fixture("tinyyolov2-7");
    let (mut engine, inputs) = load_single_input_opt(&base, "model.onnx", true);
    validate_once(&mut engine, &inputs, &base, "model.onnx", 0, 5e-3);
    c.bench_function("tinyyolov2-7-graphopt", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_tinyyolov3_11_graphopt(c: &mut Criterion) {
    let base = fixture("tiny-yolov3-11");
    let (mut engine, inputs) = load_multi_input_opt(&base, "yolov3-tiny.onnx", true);
    validate_once(&mut engine, &inputs, &base, "yolov3-tiny.onnx", 0, 5e-3);
    c.bench_function("tiny-yolov3-11-graphopt", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_yolov4_11_graphopt(c: &mut Criterion) {
    let base = fixture("yolov4-11");
    let (mut engine, inputs) = load_single_input_opt(&base, "yolov4.onnx", true);
    validate_once(&mut engine, &inputs, &base, "yolov4.onnx", 0, 5e-3);
    c.bench_function("yolov4-11-graphopt", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

// --- XNNPACK variants ---

#[cfg(feature = "xnnpack")]
fn bench_mnist12_xnnpack(c: &mut Criterion) {
    let base = fixture("mnist-12");
    let (mut engine, inputs) = load_single_input_mode(&base, "mnist-12.onnx", EngineMode::Xnnpack);
    validate_once(&mut engine, &inputs, &base, "mnist-12.onnx", 0, 1e-4);
    c.bench_function("mnist-12-xnnpack", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

#[cfg(feature = "xnnpack")]
fn bench_mobilenetv2_7_xnnpack(c: &mut Criterion) {
    let base = fixture("mobilenetv2-7");
    let (mut engine, inputs) = load_single_input_mode(&base, "mobilenetv2-7.onnx", EngineMode::Xnnpack);
    validate_once(&mut engine, &inputs, &base, "mobilenetv2-7.onnx", 0, 5e-3);
    c.bench_function("mobilenetv2-7-xnnpack", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

#[cfg(feature = "xnnpack")]
fn bench_mobilenetv2_12_xnnpack(c: &mut Criterion) {
    let base = fixture("mobilenetv2-12");
    let (mut engine, inputs) = load_single_input_mode(&base, "mobilenetv2-12.onnx", EngineMode::Xnnpack);
    validate_once(&mut engine, &inputs, &base, "mobilenetv2-12.onnx", 0, 5e-3);
    c.bench_function("mobilenetv2-12-xnnpack", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

#[cfg(feature = "xnnpack")]
fn bench_tinyyolov2_7_xnnpack(c: &mut Criterion) {
    let base = fixture("tinyyolov2-7");
    let (mut engine, inputs) = load_single_input_mode(&base, "model.onnx", EngineMode::Xnnpack);
    validate_once(&mut engine, &inputs, &base, "model.onnx", 0, 5e-3);
    c.bench_function("tinyyolov2-7-xnnpack", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

#[cfg(feature = "xnnpack")]
fn bench_tinyyolov3_11_xnnpack(c: &mut Criterion) {
    let base = fixture("tiny-yolov3-11");
    let (mut engine, inputs) = load_multi_input_mode(&base, "yolov3-tiny.onnx", EngineMode::Xnnpack);
    validate_once(&mut engine, &inputs, &base, "yolov3-tiny.onnx", 0, 5e-3);
    c.bench_function("tiny-yolov3-11-xnnpack", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

#[cfg(feature = "xnnpack")]
fn bench_yolov4_11_xnnpack(c: &mut Criterion) {
    let base = fixture("yolov4-11");
    let (mut engine, inputs) = load_single_input_mode(&base, "yolov4.onnx", EngineMode::Xnnpack);
    validate_once(&mut engine, &inputs, &base, "yolov4.onnx", 0, 5e-3);
    c.bench_function("yolov4-11-xnnpack", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

#[cfg(not(feature = "xnnpack"))]
criterion_group!(
    benches,
    bench_mnist1,
    bench_mnist7,
    bench_mnist8,
    bench_mnist12,
    bench_mobilenetv2_7,
    bench_mobilenetv2_12,
    bench_tinyyolov2_7,
    bench_tinyyolov3_11,
    bench_yolov4_11,
    bench_mnist12_graphopt,
    bench_mobilenetv2_7_graphopt,
    bench_mobilenetv2_12_graphopt,
    bench_tinyyolov2_7_graphopt,
    bench_tinyyolov3_11_graphopt,
    bench_yolov4_11_graphopt,
);

#[cfg(feature = "xnnpack")]
criterion_group!(
    benches,
    bench_mnist1,
    bench_mnist7,
    bench_mnist8,
    bench_mnist12,
    bench_mobilenetv2_7,
    bench_mobilenetv2_12,
    bench_tinyyolov2_7,
    bench_tinyyolov3_11,
    bench_yolov4_11,
    bench_mnist12_graphopt,
    bench_mobilenetv2_7_graphopt,
    bench_mobilenetv2_12_graphopt,
    bench_tinyyolov2_7_graphopt,
    bench_tinyyolov3_11_graphopt,
    bench_yolov4_11_graphopt,
    bench_mnist12_xnnpack,
    bench_mobilenetv2_7_xnnpack,
    bench_mobilenetv2_12_xnnpack,
    bench_tinyyolov2_7_xnnpack,
    bench_tinyyolov3_11_xnnpack,
    bench_yolov4_11_xnnpack,
);

criterion_main!(benches);
