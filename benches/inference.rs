use std::collections::HashMap;
use std::fs;
use std::path::Path;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use onnx_infer::InferenceEngine;
use onnx_infer::Tensor;
use prost::Message;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn load_single_input(base: &Path, model_file: &str) -> (InferenceEngine, HashMap<String, Tensor>) {
    let model_bytes = fs::read(base.join(model_file)).expect("read model");

    let model = onnx_infer::onnx::ModelProto::decode(&model_bytes[..]).unwrap();
    let graph = model.graph.as_ref().unwrap();
    let input_name = graph.input[0].name.clone();

    let test_dir = base.join("test_data_set_0");
    let input_bytes = fs::read(test_dir.join("input_0.pb")).expect("read input");
    let input = Tensor::from_proto_bytes(&input_bytes).expect("parse input");

    let engine = InferenceEngine::new(&model_bytes).expect("load model");

    let mut inputs = HashMap::new();
    inputs.insert(input_name, input);
    (engine, inputs)
}

fn load_multi_input(base: &Path, model_file: &str) -> (InferenceEngine, HashMap<String, Tensor>) {
    let model_bytes = fs::read(base.join(model_file)).expect("read model");

    let model = onnx_infer::onnx::ModelProto::decode(&model_bytes[..]).unwrap();
    let graph = model.graph.as_ref().unwrap();

    let test_dir = base.join("test_data_set_0");
    let mut inputs = HashMap::new();
    for i in 0..graph.input.len() {
        let pb_path = test_dir.join(format!("input_{i}.pb"));
        if pb_path.exists() {
            let input = Tensor::from_proto_bytes(&fs::read(&pb_path).expect("read input"))
                .expect("parse input");
            inputs.insert(graph.input[i].name.clone(), input);
        }
    }

    let engine = InferenceEngine::new(&model_bytes).expect("load model");
    (engine, inputs)
}

fn bench_mnist1(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("mnist-1"), "model.onnx");
    c.bench_function("mnist-1", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mnist7(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("mnist-7"), "model.onnx");
    c.bench_function("mnist-7", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mnist8(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("mnist-8"), "model.onnx");
    c.bench_function("mnist-8", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mnist12(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("mnist-12"), "mnist-12.onnx");
    c.bench_function("mnist-12", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mobilenetv2_7(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("mobilenetv2-7"), "mobilenetv2-7.onnx");
    c.bench_function("mobilenetv2-7", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_mobilenetv2_12(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("mobilenetv2-12"), "mobilenetv2-12.onnx");
    c.bench_function("mobilenetv2-12", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_tinyyolov2_7(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("tinyyolov2-7"), "model.onnx");
    c.bench_function("tinyyolov2-7", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_tinyyolov3_11(c: &mut Criterion) {
    let (mut engine, inputs) = load_multi_input(&fixture("tiny-yolov3-11"), "yolov3-tiny.onnx");
    c.bench_function("tiny-yolov3-11", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

fn bench_yolov4_11(c: &mut Criterion) {
    let (mut engine, inputs) = load_single_input(&fixture("yolov4-11"), "yolov4.onnx");
    c.bench_function("yolov4-11", |b| {
        b.iter(|| engine.run(inputs.clone()).unwrap())
    });
}

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
);
criterion_main!(benches);
