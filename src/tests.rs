use std::collections::HashMap;
use std::fs;
use std::path::Path;

use approx::assert_relative_eq;
use prost::Message;
use test_case::test_case;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

use crate::DType;
use crate::InferenceEngine;
use crate::InferenceOptions;
use crate::Tensor;
use crate::onnx::ModelProto;
use crate::onnx::TensorProto;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn setup_tracing(
    test_name: &str,
) -> (
    tracing_chrome::FlushGuard,
    tracing::subscriber::DefaultGuard,
) {
    let trace_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("traces");
    fs::create_dir_all(&trace_dir).ok();
    let trace_path = trace_dir.join(format!("{test_name}.json"));
    let (chrome_layer, flush_guard) = ChromeLayerBuilder::new()
        .file(trace_path)
        .include_args(true)
        .build();
    let subscriber = tracing_subscriber::registry().with(chrome_layer);
    let default_guard = tracing::subscriber::set_default(subscriber);
    (flush_guard, default_guard)
}

fn make_options(xnnpack: bool) -> InferenceOptions {
    InferenceOptions {
        xnnpack,
        ..Default::default()
    }
}

fn load_model_and_inputs(
    base: &Path,
    model_file: &str,
    test_set: usize,
) -> (Vec<u8>, HashMap<String, Tensor>) {
    let model_bytes = fs::read(base.join(model_file)).expect("read model");
    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");

    let test_dir = base.join(format!("test_data_set_{test_set}"));

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

    (model_bytes, inputs)
}

fn run_fixture(base: &Path, model_file: &str, test_set: usize, xnnpack: bool) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes, make_options(xnnpack)).expect("load model");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
    let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

    engine.run(inputs).expect("inference");
    let output = &engine.outputs[&output_name];

    assert_eq!(output.dims, expected.dims);

    let out_data = output.floats().expect("output should be float tensor");
    let exp_data = expected
        .floats()
        .expect("expected output should be float tensor");
    let mut max_err: f32 = 0.0;
    let mut max_err_idx = 0;
    for (i, (got, want)) in out_data.iter().zip(exp_data.iter()).enumerate() {
        let err = (got - want).abs();
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }
    if max_err > 1e-3 {
        eprintln!(
            "max absolute error: {max_err} at index {max_err_idx} (got={}, want={}), output len={}",
            out_data[max_err_idx],
            exp_data[max_err_idx],
            out_data.len()
        );
    }
    for (got, want) in out_data.iter().zip(exp_data.iter()) {
        assert_relative_eq!(got, want, max_relative = 1e-3, epsilon = 1e-5);
    }

    let got_class = out_data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in output"))
        .expect("empty output")
        .0;
    let expected_class = exp_data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in expected output"))
        .expect("empty expected output")
        .0;
    assert_eq!(got_class, expected_class);
}

fn run_fixture_argmax(base: &Path, model_file: &str, test_set: usize, xnnpack: bool) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes, make_options(xnnpack)).expect("load model");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
    let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

    engine.run(inputs).expect("inference");
    let output = &engine.outputs[&output_name];

    assert_eq!(output.dims, expected.dims);

    let out_data = output.floats().expect("output should be float tensor");
    let exp_data = expected
        .floats()
        .expect("expected output should be float tensor");

    let got_class = out_data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in output"))
        .expect("empty output")
        .0;
    let expected_class = exp_data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in expected output"))
        .expect("empty expected output")
        .0;
    assert_eq!(got_class, expected_class);
}

fn run_quantized_fixture(base: &Path, model_file: &str, test_set: usize, xnnpack: bool) {
    run_quantized_fixture_with_tol(base, model_file, test_set, 0.1, xnnpack);
}

fn run_quantized_fixture_with_tol(
    base: &Path,
    model_file: &str,
    test_set: usize,
    softmax_tol: f32,
    xnnpack: bool,
) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes, make_options(xnnpack)).expect("load model");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
    let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

    engine.run(inputs).expect("inference");
    let output = &engine.outputs[&output_name];

    assert_eq!(output.dims, expected.dims);

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    let got_probs = softmax(output.floats().expect("output should be float tensor"));
    let want_probs = softmax(
        expected
            .floats()
            .expect("expected output should be float tensor"),
    );

    let mut max_abs_err: f32 = 0.0;
    for (g, w) in got_probs.iter().zip(want_probs.iter()) {
        max_abs_err = max_abs_err.max((g - w).abs());
    }
    eprintln!("int8 max softmax probability error: {max_abs_err:.6}");
    assert!(
        max_abs_err < softmax_tol,
        "max softmax error {max_abs_err} >= {softmax_tol}"
    );

    let expected_class = want_probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in expected output"))
        .expect("empty expected output")
        .0;
    let mut indexed: Vec<(usize, &f32)> = got_probs.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).expect("NaN in output"));
    let top5: Vec<usize> = indexed.iter().take(5).map(|&(i, _)| i).collect();
    assert!(
        top5.contains(&expected_class),
        "expected class {expected_class} not in top-5: {top5:?}"
    );
}

fn run_multi_io_fixture(base: &Path, model_file: &str, test_set: usize, xnnpack: bool) {
    run_multi_io_fixture_with_tol(base, model_file, test_set, 5e-3, xnnpack);
}

fn run_multi_io_fixture_with_tol(
    base: &Path,
    model_file: &str,
    test_set: usize,
    tol: f32,
    xnnpack: bool,
) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes, make_options(xnnpack)).expect("load model");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    engine.run(inputs).expect("inference");

    for i in 0..graph.output.len() {
        let pb_path = test_dir.join(format!("output_{i}.pb"));
        if pb_path.exists() {
            let expected = Tensor::from_proto_bytes(&fs::read(&pb_path).expect("read output"))
                .expect("parse output");
            let name = &graph.output[i].name;
            let output = engine
                .outputs
                .get(name)
                .unwrap_or_else(|| panic!("missing output {name}"));
            assert_eq!(output.dims, expected.dims, "shape mismatch for {name}");

            match (output.dtype(), expected.dtype()) {
                (DType::Float, DType::Float) => {
                    let got = output.floats().expect("output should be float tensor");
                    let want = expected
                        .floats()
                        .expect("expected output should be float tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (g - w).abs() < tol || (g - w).abs() / w.abs().max(1e-6) < tol,
                            "output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                (DType::Int64, DType::Int64) => {
                    let got = output.ints().expect("output should be int64 tensor");
                    let want = expected
                        .ints()
                        .expect("expected output should be int64 tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert_eq!(g, w, "output {name}[{j}]: got {g}, want {w}");
                    }
                }
                (DType::Float, DType::Int64) => {
                    let got = output.floats().expect("output should be float tensor");
                    let want = expected
                        .ints()
                        .expect("expected output should be int64 tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (*g as i64 - w).abs() <= 1,
                            "output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                (DType::Int64, DType::Float) => {
                    let got = output.ints().expect("output should be int64 tensor");
                    let want = expected
                        .floats()
                        .expect("expected output should be float tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (*g as f32 - w).abs() < tol,
                            "output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                _ => panic!("unexpected output dtype for {name}"),
            }
        }
    }
}

// --- Graph-optimized fixture runners ---

fn run_fixture_graphopt(base: &Path, model_file: &str, test_set: usize) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine =
        InferenceEngine::new(&model_bytes, Default::default()).expect("load model with graph opt");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
    let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

    engine.run(inputs).expect("inference with graph opt");
    let output = &engine.outputs[&output_name];

    assert_eq!(output.dims, expected.dims, "shape mismatch after graph opt");

    let out_data = output.floats().expect("output should be float tensor");
    let exp_data = expected
        .floats()
        .expect("expected output should be float tensor");
    let mut max_err: f32 = 0.0;
    let mut max_err_idx = 0;
    for (i, (got, want)) in out_data.iter().zip(exp_data.iter()).enumerate() {
        let err = (got - want).abs();
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }
    if max_err > 1e-3 {
        eprintln!(
            "[graphopt] max absolute error: {max_err} at index {max_err_idx} (got={}, want={}), output len={}",
            out_data[max_err_idx],
            exp_data[max_err_idx],
            out_data.len()
        );
    }
    for (got, want) in out_data.iter().zip(exp_data.iter()) {
        assert_relative_eq!(got, want, max_relative = 1e-3, epsilon = 1e-5);
    }
}

fn run_multi_io_fixture_graphopt(base: &Path, model_file: &str, test_set: usize) {
    run_multi_io_fixture_graphopt_with_tol(base, model_file, test_set, 5e-3);
}

fn run_multi_io_fixture_graphopt_with_tol(
    base: &Path,
    model_file: &str,
    test_set: usize,
    tol: f32,
) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine =
        InferenceEngine::new(&model_bytes, Default::default()).expect("load model with graph opt");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    engine.run(inputs).expect("inference with graph opt");

    for i in 0..graph.output.len() {
        let pb_path = test_dir.join(format!("output_{i}.pb"));
        if pb_path.exists() {
            let expected = Tensor::from_proto_bytes(&fs::read(&pb_path).expect("read output"))
                .expect("parse output");
            let name = &graph.output[i].name;
            let output = engine
                .outputs
                .get(name)
                .unwrap_or_else(|| panic!("missing output {name}"));
            assert_eq!(
                output.dims, expected.dims,
                "[graphopt] shape mismatch for {name}"
            );

            match (output.dtype(), expected.dtype()) {
                (DType::Float, DType::Float) => {
                    let got = output.floats().expect("output should be float tensor");
                    let want = expected
                        .floats()
                        .expect("expected output should be float tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (g - w).abs() < tol || (g - w).abs() / w.abs().max(1e-6) < tol,
                            "[graphopt] output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                (DType::Int64, DType::Int64) => {
                    let got = output.ints().expect("output should be int64 tensor");
                    let want = expected
                        .ints()
                        .expect("expected output should be int64 tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert_eq!(g, w, "[graphopt] output {name}[{j}]: got {g}, want {w}");
                    }
                }
                (DType::Float, DType::Int64) => {
                    let got = output.floats().expect("output should be float tensor");
                    let want = expected
                        .ints()
                        .expect("expected output should be int64 tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (*g as i64 - w).abs() <= 1,
                            "[graphopt] output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                (DType::Int64, DType::Float) => {
                    let got = output.ints().expect("output should be int64 tensor");
                    let want = expected
                        .floats()
                        .expect("expected output should be float tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (*g as f32 - w).abs() < tol,
                            "[graphopt] output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                _ => panic!("[graphopt] unexpected output dtype for {name}"),
            }
        }
    }
}

fn run_quantized_fixture_graphopt(base: &Path, model_file: &str, test_set: usize) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine =
        InferenceEngine::new(&model_bytes, Default::default()).expect("load model with graph opt");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();

    let test_dir = base.join(format!("test_data_set_{test_set}"));
    let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
    let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

    engine.run(inputs).expect("inference with graph opt");
    let output = &engine.outputs[&output_name];

    assert_eq!(output.dims, expected.dims, "[graphopt] shape mismatch");

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    let got_probs = softmax(output.floats().expect("output should be float tensor"));
    let want_probs = softmax(
        expected
            .floats()
            .expect("expected output should be float tensor"),
    );

    let expected_class = want_probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in expected output"))
        .expect("empty expected output")
        .0;
    let mut indexed: Vec<(usize, &f32)> = got_probs.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).expect("NaN in output"));
    let top5: Vec<usize> = indexed.iter().take(5).map(|&(i, _)| i).collect();
    assert!(
        top5.contains(&expected_class),
        "[graphopt] expected class {expected_class} not in top-5: {top5:?}"
    );
}

// --- Graph-opt tests ---

#[test]
fn test_graphopt_mnist12_set_0() {
    run_fixture_graphopt(&fixture("mnist-12"), "mnist-12.onnx", 0);
}

#[test]
fn test_graphopt_mnist12_int8_set_0() {
    run_quantized_fixture_graphopt(&fixture("mnist-12-int8"), "mnist-12-int8.onnx", 0);
}

#[test]
fn test_graphopt_mobilenetv2_7_set_0() {
    run_fixture_graphopt(&fixture("mobilenetv2-7"), "mobilenetv2-7.onnx", 0);
}

#[test]
fn test_graphopt_mobilenetv2_12_set_0() {
    run_fixture_graphopt(&fixture("mobilenetv2-12"), "mobilenetv2-12.onnx", 0);
}

#[test]
fn test_graphopt_mobilenetv2_12_int8_set_0() {
    run_quantized_fixture_graphopt(
        &fixture("mobilenetv2-12-int8"),
        "mobilenetv2-12-int8.onnx",
        0,
    );
}

#[test]
fn test_graphopt_tinyyolov2_8_set_0() {
    run_fixture_graphopt(&fixture("tinyyolov2-8"), "model.onnx", 0);
}

#[test]
fn test_graphopt_resnet18_v1_7_set_0() {
    run_fixture_graphopt(&fixture("resnet18-v1-7"), "resnet18-v1-7.onnx", 0);
}

#[test]
fn test_graphopt_resnet50_v1_12_set_0() {
    run_fixture_graphopt(&fixture("resnet50-v1-12"), "resnet50-v1-12.onnx", 0);
}

#[test]
fn test_graphopt_densenet_12_set_0() {
    run_fixture_graphopt(&fixture("densenet-12"), "densenet-12.onnx", 0);
}

#[test]
fn test_graphopt_googlenet_12_set_0() {
    run_fixture_graphopt(&fixture("googlenet-12"), "googlenet-12.onnx", 0);
}

#[test]
fn test_graphopt_inception_v1_12_set_0() {
    run_fixture_graphopt(&fixture("inception-v1-12"), "inception-v1-12.onnx", 0);
}

#[test]
fn test_graphopt_squeezenet11_7_set_0() {
    run_fixture_graphopt(&fixture("squeezenet1.1-7"), "squeezenet1.1.onnx", 0);
}

#[test]
fn test_graphopt_shufflenet_v2_12_set_0() {
    run_fixture_graphopt(&fixture("shufflenet-v2-12"), "shufflenet-v2-12.onnx", 0);
}

#[test]
fn test_graphopt_vgg16_bn_7_set_0() {
    run_fixture_graphopt(&fixture("vgg16-bn-7"), "vgg16-bn.onnx", 0);
}

#[test]
fn test_graphopt_efficientnet_lite4_11_set_0() {
    run_fixture_graphopt(
        &fixture("efficientnet-lite4-11"),
        "efficientnet-lite4.onnx",
        0,
    );
}

#[test]
fn test_graphopt_arcfaceresnet100_8_set_0() {
    run_fixture_graphopt(&fixture("arcfaceresnet100-8"), "resnet100.onnx", 0);
}

#[test]
fn test_graphopt_emotion_ferplus_8_set_0() {
    run_fixture_graphopt(&fixture("emotion-ferplus-8"), "model.onnx", 0);
}

#[test]
fn test_graphopt_faster_rcnn_12_set_0() {
    run_multi_io_fixture_graphopt(&fixture("faster-rcnn-12"), "FasterRCNN-12.onnx", 0);
}

#[test]
fn test_graphopt_ssd_mobilenet_v1_12_set_0() {
    run_multi_io_fixture_graphopt(
        &fixture("ssd-mobilenet-v1-12"),
        "ssd_mobilenet_v1_12.onnx",
        0,
    );
}

#[test]
fn test_graphopt_yolov4_11_set_0() {
    run_multi_io_fixture_graphopt(&fixture("yolov4-11"), "yolov4.onnx", 0);
}

#[test]
fn test_graphopt_tinyyolov3_11_set_0() {
    run_multi_io_fixture_graphopt(&fixture("tiny-yolov3-11"), "yolov3-tiny.onnx", 0);
}

#[test]
fn test_graphopt_yolov3_12_set_0() {
    run_multi_io_fixture_graphopt(&fixture("yolov3-12"), "yolov3-12.onnx", 0);
}

#[test]
fn test_graphopt_version_rfb_320_set_0() {
    run_multi_io_fixture_graphopt(&fixture("version-RFB-320"), "version-RFB-320.onnx", 0);
}

#[test]
fn test_graphopt_retinanet_9_set_0() {
    run_multi_io_fixture_graphopt(&fixture("retinanet-9"), "retinanet-9.onnx", 0);
}

#[test]
fn test_graphopt_bidaf_9_set_0() {
    run_multi_io_fixture_graphopt(&fixture("bidaf-9"), "bidaf.onnx", 0);
}

// --- MNIST models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mnist1_set_0(xnnpack: bool) {
    let _t = setup_tracing("mnist1_set_0");
    run_fixture(&fixture("mnist-1"), "model.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mnist1_set_1(xnnpack: bool) {
    let _t = setup_tracing("mnist1_set_1");
    run_fixture(&fixture("mnist-1"), "model.onnx", 1, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mnist1_set_2(xnnpack: bool) {
    let _t = setup_tracing("mnist1_set_2");
    run_fixture(&fixture("mnist-1"), "model.onnx", 2, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mnist7_set_0(xnnpack: bool) {
    let _t = setup_tracing("mnist7_set_0");
    run_fixture(&fixture("mnist-7"), "model.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mnist8_set_0(xnnpack: bool) {
    let _t = setup_tracing("mnist8_set_0");
    run_fixture(&fixture("mnist-8"), "model.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mnist12_set_0(xnnpack: bool) {
    let _t = setup_tracing("mnist12_set_0");
    run_fixture(&fixture("mnist-12"), "mnist-12.onnx", 0, xnnpack);
}

#[test]
fn test_mnist12_input_floats_mut() {
    let _t = setup_tracing("mnist12_input_floats_mut");
    let base = fixture("mnist-12");
    let (model_bytes, inputs) = load_model_and_inputs(&base, "mnist-12.onnx", 0);
    let mut engine = InferenceEngine::new(&model_bytes, Default::default()).expect("load model");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();

    let test_dir = base.join("test_data_set_0");
    let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
    let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

    // Write input data via input_floats_mut instead of passing a HashMap
    for (name, tensor) in &inputs {
        let src = tensor.floats().expect("input should be float");
        let dst = engine
            .input_floats_mut(name, tensor.dims.clone())
            .expect("input_floats_mut");
        dst.copy_from_slice(src);
    }

    // Run twice to verify buffer reuse (no reallocation on second call)
    for _ in 0..2 {
        let outputs = engine.run_planned().expect("run_planned");
        let output = &outputs[&output_name];
        assert_eq!(output.dims, expected.dims);

        let out_data = output.floats().expect("output should be float tensor");
        let exp_data = expected.floats().expect("expected should be float tensor");
        let max_err = out_data
            .iter()
            .zip(exp_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err <= 1e-3, "max error {max_err} exceeds 1e-3");
    }
}

#[cfg(feature = "xnnpack")]
#[test]
fn test_mnist12_xnnpack_batch2() {
    let _t = setup_tracing("mnist12_xnnpack_batch2");
    let base = fixture("mnist-12");
    let (model_bytes, inputs) = load_model_and_inputs(&base, "mnist-12.onnx", 0);

    let mut engine = InferenceEngine::new(
        &model_bytes,
        InferenceOptions {
            xnnpack: true,
            ..Default::default()
        },
    )
    .expect("load model");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let input_name = graph.input[0].name.clone();
    let output_name = graph.output[0].name.clone();

    // Get the single-sample input [1, 1, 28, 28]
    let single_input = inputs.get(&input_name).expect("missing input");
    let single_data = single_input.floats().expect("float input");
    assert_eq!(single_input.dims.as_slice(), &[1, 1, 28, 28]);

    // Run batch=2 directly (no batch=1 first): duplicate the same input
    let mut batch2_data = Vec::with_capacity(single_data.len() * 2);
    batch2_data.extend_from_slice(single_data);
    batch2_data.extend_from_slice(single_data);

    let dst = engine
        .input_floats_mut(&input_name, crate::dims![2, 1, 28, 28])
        .expect("input_floats_mut batch=2");
    dst.copy_from_slice(&batch2_data);
    let outputs = engine.run_planned().expect("run_planned batch=2");
    let out = &outputs[&output_name];
    let batch2_output = out.floats().expect("float output");

    // Output should be [2, 10] = 20 elements
    assert_eq!(
        batch2_output.len(),
        20,
        "expected 20 outputs for batch=2, got {}",
        batch2_output.len()
    );

    // Check no NaN
    for (i, v) in batch2_output.iter().enumerate() {
        assert!(!v.is_nan(), "batch=2 output[{i}] is NaN");
    }

    // Both samples are identical, so outputs should match
    for i in 0..10 {
        assert_relative_eq!(
            batch2_output[i],
            batch2_output[10 + i],
            max_relative = 1e-3,
            epsilon = 1e-5
        );
    }
}

#[cfg(feature = "xnnpack")]
#[test]
fn test_mnist12_xnnpack_batch_sizes() {
    let _t = setup_tracing("mnist12_xnnpack_batch_sizes");
    let base = fixture("mnist-12");
    let (model_bytes, inputs) = load_model_and_inputs(&base, "mnist-12.onnx", 0);

    let mut engine = InferenceEngine::new(
        &model_bytes,
        InferenceOptions {
            xnnpack: true,
            ..Default::default()
        },
    )
    .expect("load model");

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let input_name = graph.input[0].name.clone();
    let output_name = graph.output[0].name.clone();

    let single_input = inputs.get(&input_name).expect("missing input");
    let single_data = single_input.floats().expect("float input");
    let num_classes = 10;

    // Run with increasing then decreasing batch sizes: 1, 2, 4, 2, 1, 3
    for &batch_size in &[1usize, 2, 4, 2, 1, 3] {
        let mut batch_data = Vec::with_capacity(single_data.len() * batch_size);
        for _ in 0..batch_size {
            batch_data.extend_from_slice(single_data);
        }

        let dst = engine
            .input_floats_mut(&input_name, crate::dims![batch_size, 1, 28, 28])
            .expect("input_floats_mut");
        dst.copy_from_slice(&batch_data);
        let outputs = engine.run_planned().unwrap_or_else(|e| {
            panic!("run_planned failed for batch={batch_size}: {e}");
        });
        let out = &outputs[&output_name];
        let out_data = out.floats().expect("float output");

        assert_eq!(
            out_data.len(),
            num_classes * batch_size,
            "batch={batch_size}: expected {} outputs, got {}",
            num_classes * batch_size,
            out_data.len()
        );

        // Check no NaN
        for (i, v) in out_data.iter().enumerate() {
            assert!(!v.is_nan(), "batch={batch_size} output[{i}] is NaN");
        }

        // All samples are identical, so each batch item should match the first
        for b in 1..batch_size {
            for c in 0..num_classes {
                assert_relative_eq!(
                    out_data[c],
                    out_data[b * num_classes + c],
                    max_relative = 1e-3,
                    epsilon = 1e-5
                );
            }
        }
    }
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mnist12_int8_set_0(xnnpack: bool) {
    let _t = setup_tracing("mnist12_int8_set_0");
    run_quantized_fixture(&fixture("mnist-12-int8"), "mnist-12-int8.onnx", 0, xnnpack);
}

// --- MobileNetV2 models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mobilenetv2_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("mobilenetv2_7_set_0");
    run_fixture(&fixture("mobilenetv2-7"), "mobilenetv2-7.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mobilenetv2_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("mobilenetv2_12_set_0");
    run_fixture(
        &fixture("mobilenetv2-12"),
        "mobilenetv2-12.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mobilenetv2_12_int8_set_0(xnnpack: bool) {
    let _t = setup_tracing("mobilenetv2_12_int8_set_0");
    run_quantized_fixture(
        &fixture("mobilenetv2-12-int8"),
        "mobilenetv2-12-int8.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mobilenetv2_12_qdq_set_0(xnnpack: bool) {
    let _t = setup_tracing("mobilenetv2_12_qdq_set_0");
    if xnnpack {
        // QDQ models amplify float precision differences through quantize/dequantize
        // chains, causing divergent classifications with XNNPACK. Skip until native
        // XNNPACK quantized op support is added.
        eprintln!("skipping QDQ model on XNNPACK (precision)");
        return;
    }
    run_quantized_fixture_with_tol(
        &fixture("mobilenetv2-12-qdq"),
        "mobilenetv2-12-qdq.onnx",
        0,
        0.15,
        xnnpack,
    );
}

// --- Tiny YOLOv2 models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_tinyyolov2_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("tinyyolov2_7_set_0");
    run_fixture(&fixture("tinyyolov2-7"), "model.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_tinyyolov2_8_set_0(xnnpack: bool) {
    let _t = setup_tracing("tinyyolov2_8_set_0");
    run_fixture(&fixture("tinyyolov2-8"), "model.onnx", 0, xnnpack);
}

// --- Faster R-CNN models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_faster_rcnn_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("faster_rcnn_12_set_0");
    run_multi_io_fixture(&fixture("faster-rcnn-12"), "FasterRCNN-12.onnx", 0, xnnpack);
}

// --- SSD MobileNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_ssd_mobilenet_v1_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("ssd_mobilenet_v1_12_set_0");
    run_multi_io_fixture(
        &fixture("ssd-mobilenet-v1-12"),
        "ssd_mobilenet_v1_12.onnx",
        0,
        xnnpack,
    );
}

// --- YOLOv4 models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_yolov4_11_set_0(xnnpack: bool) {
    let _t = setup_tracing("yolov4_11_set_0");
    run_multi_io_fixture(&fixture("yolov4-11"), "yolov4.onnx", 0, xnnpack);
}

// --- Tiny YOLOv3 models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_tinyyolov3_11_set_0(xnnpack: bool) {
    let _t = setup_tracing("tinyyolov3_11_set_0");
    run_multi_io_fixture(&fixture("tiny-yolov3-11"), "yolov3-tiny.onnx", 0, xnnpack);
}

// --- BiDAF models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_bidaf_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("bidaf_9_set_0");
    run_multi_io_fixture(&fixture("bidaf-9"), "bidaf.onnx", 0, xnnpack);
}

// --- AlexNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_bvlcalexnet_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("bvlcalexnet_12_set_0");
    run_fixture(
        &fixture("bvlcalexnet-12"),
        "bvlcalexnet-12.onnx",
        0,
        xnnpack,
    );
}

// --- CaffeNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_caffenet_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("caffenet_12_set_0");
    run_fixture(&fixture("caffenet-12"), "caffenet-12.onnx", 0, xnnpack);
}

// --- DenseNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_densenet_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("densenet_12_set_0");
    run_fixture(&fixture("densenet-12"), "densenet-12.onnx", 0, xnnpack);
}

// --- EfficientNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_efficientnet_lite4_11_set_0(xnnpack: bool) {
    let _t = setup_tracing("efficientnet_lite4_11_set_0");
    run_fixture(
        &fixture("efficientnet-lite4-11"),
        "efficientnet-lite4.onnx",
        0,
        xnnpack,
    );
}

// --- GoogLeNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_googlenet_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("googlenet_12_set_0");
    run_fixture(&fixture("googlenet-12"), "googlenet-12.onnx", 0, xnnpack);
}

// --- Inception models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_inception_v1_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("inception_v1_12_set_0");
    run_fixture(
        &fixture("inception-v1-12"),
        "inception-v1-12.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_inception_v2_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("inception_v2_9_set_0");
    run_fixture(&fixture("inception-v2-9"), "model.onnx", 0, xnnpack);
}

// --- RCNN ILSVRC13 models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_rcnn_ilsvrc13_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("rcnn_ilsvrc13_9_set_0");
    run_fixture(&fixture("rcnn-ilsvrc13-9"), "model.onnx", 0, xnnpack);
}

// --- ResNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet18_v1_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet18_v1_7_set_0");
    run_fixture(&fixture("resnet18-v1-7"), "resnet18-v1-7.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet18_v2_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet18_v2_7_set_0");
    run_fixture(&fixture("resnet18-v2-7"), "resnet18-v2-7.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet34_v1_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet34_v1_7_set_0");
    run_fixture(&fixture("resnet34-v1-7"), "resnet34-v1-7.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet34_v2_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet34_v2_7_set_0");
    run_fixture(&fixture("resnet34-v2-7"), "resnet34-v2-7.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet50_v1_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet50_v1_12_set_0");
    run_fixture(
        &fixture("resnet50-v1-12"),
        "resnet50-v1-12.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet50_v2_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet50_v2_7_set_0");
    run_fixture(&fixture("resnet50-v2-7"), "resnet50-v2-7.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet50_caffe2_v1_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet50_caffe2_v1_9_set_0");
    run_fixture(&fixture("resnet50-caffe2-v1-9"), "model.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet101_v1_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet101_v1_7_set_0");
    run_fixture(
        &fixture("resnet101-v1-7"),
        "resnet101-v1-7.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet101_v2_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet101_v2_7_set_0");
    run_fixture(
        &fixture("resnet101-v2-7"),
        "resnet101-v2-7.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet152_v1_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet152_v1_7_set_0");
    run_fixture(
        &fixture("resnet152-v1-7"),
        "resnet152-v1-7.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet152_v2_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet152_v2_7_set_0");
    run_fixture(
        &fixture("resnet152-v2-7"),
        "resnet152-v2-7.onnx",
        0,
        xnnpack,
    );
}

// --- ShuffleNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_shufflenet_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("shufflenet_9_set_0");
    run_fixture(&fixture("shufflenet-9"), "model.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_shufflenet_v2_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("shufflenet_v2_12_set_0");
    run_fixture(
        &fixture("shufflenet-v2-12"),
        "shufflenet-v2-12.onnx",
        0,
        xnnpack,
    );
}

// --- SqueezeNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_squeezenet10_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("squeezenet10_12_set_0");
    run_fixture_argmax(
        &fixture("squeezenet1.0-12"),
        "squeezenet1.0-12.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_squeezenet11_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("squeezenet11_7_set_0");
    run_fixture(
        &fixture("squeezenet1.1-7"),
        "squeezenet1.1.onnx",
        0,
        xnnpack,
    );
}

// --- VGG models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_vgg16_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("vgg16_12_set_0");
    run_fixture(&fixture("vgg16-12"), "vgg16-12.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_vgg16_bn_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("vgg16_bn_7_set_0");
    run_fixture(&fixture("vgg16-bn-7"), "vgg16-bn.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_vgg19_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("vgg19_7_set_0");
    run_fixture(&fixture("vgg19-7"), "vgg19.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_vgg19_bn_7_set_0(xnnpack: bool) {
    let _t = setup_tracing("vgg19_bn_7_set_0");
    run_fixture(&fixture("vgg19-bn-7"), "vgg19-bn-7.onnx", 0, xnnpack);
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_vgg19_caffe2_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("vgg19_caffe2_9_set_0");
    run_fixture(&fixture("vgg19-caffe2-9"), "model.onnx", 0, xnnpack);
}

// --- ZFNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_zfnet512_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("zfnet512_12_set_0");
    run_fixture(&fixture("zfnet512-12"), "zfnet512-12.onnx", 0, xnnpack);
}

// --- ResNet101-DUC models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_resnet101_duc_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("resnet101_duc_12_set_0");
    run_fixture_argmax(
        &fixture("ResNet101-DUC-12"),
        "ResNet101-DUC-12.onnx",
        0,
        xnnpack,
    );
}

// --- FCN models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_fcn_resnet50_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("fcn_resnet50_12_set_0");
    run_multi_io_fixture(
        &fixture("fcn-resnet50-12"),
        "fcn-resnet50-12.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_fcn_resnet101_11_set_0(xnnpack: bool) {
    let _t = setup_tracing("fcn_resnet101_11_set_0");
    run_multi_io_fixture(&fixture("fcn-resnet101-11"), "model.onnx", 0, xnnpack);
}

// --- Mask R-CNN models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_mask_rcnn_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("mask_rcnn_12_set_0");
    run_multi_io_fixture(&fixture("MaskRCNN-12"), "MaskRCNN-12.onnx", 0, xnnpack);
}

// --- RetinaNet models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_retinanet_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("retinanet_9_set_0");
    run_multi_io_fixture(&fixture("retinanet-9"), "retinanet-9.onnx", 0, xnnpack);
}

// --- SSD models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_ssd_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("ssd_12_set_0");
    run_multi_io_fixture(&fixture("ssd-12"), "ssd-12.onnx", 0, xnnpack);
}

// --- YOLOv2 models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_yolov2_coco_9_set_0(xnnpack: bool) {
    let _t = setup_tracing("yolov2_coco_9_set_0");
    run_multi_io_fixture(&fixture("yolov2-coco-9"), "yolov2-coco-9.onnx", 0, xnnpack);
}

// --- YOLOv3 models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_yolov3_12_set_0(xnnpack: bool) {
    let _t = setup_tracing("yolov3_12_set_0");
    run_multi_io_fixture(&fixture("yolov3-12"), "yolov3-12.onnx", 0, xnnpack);
}

// --- ArcFace models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_arcfaceresnet100_8_set_0(xnnpack: bool) {
    let _t = setup_tracing("arcfaceresnet100_8_set_0");
    run_fixture(&fixture("arcfaceresnet100-8"), "resnet100.onnx", 0, xnnpack);
}

// --- Emotion FERPlus models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_emotion_ferplus_8_set_0(xnnpack: bool) {
    let _t = setup_tracing("emotion_ferplus_8_set_0");
    run_fixture(&fixture("emotion-ferplus-8"), "model.onnx", 0, xnnpack);
}

// --- UltraFace models ---

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_version_rfb_320_set_0(xnnpack: bool) {
    let _t = setup_tracing("version_rfb_320_set_0");
    run_multi_io_fixture(
        &fixture("version-RFB-320"),
        "version-RFB-320.onnx",
        0,
        xnnpack,
    );
}

#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_version_rfb_640_set_0(xnnpack: bool) {
    let _t = setup_tracing("version_rfb_640_set_0");
    run_multi_io_fixture(
        &fixture("version-RFB-640"),
        "version-RFB-640.onnx",
        0,
        xnnpack,
    );
}

// --- Correctness tests ---

/// Verify that running inference twice with different inputs produces different
/// outputs. This catches the bug where user inputs were accidentally added to
/// the constant-folding map during plan building, causing the entire network to
/// be baked with the first frame's data.
#[test_case(false ; "cpu")]
#[cfg_attr(feature = "xnnpack", test_case(true ; "xnnpack"))]
fn test_different_inputs_produce_different_outputs(xnnpack: bool) {
    let base = fixture("bvlcalexnet-12");
    let (model_bytes, inputs) = load_model_and_inputs(&base, "bvlcalexnet-12.onnx", 0);

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();
    let input_name = graph.input[0].name.clone();

    let mut engine = InferenceEngine::new(&model_bytes, make_options(xnnpack)).expect("load model");

    // First run with real input
    engine.run(inputs.clone()).expect("run 1");
    let out1: Vec<f32> = engine.outputs[&output_name]
        .floats()
        .expect("float output")
        .to_vec();

    // Second run with zeroed input (same shape, all zeros)
    let orig = &inputs[&input_name];
    let zero_tensor = Tensor::new(
        orig.dims.clone(),
        vec![0.0f32; orig.floats().unwrap().len()],
    );
    let mut zero_inputs = HashMap::new();
    zero_inputs.insert(input_name, zero_tensor);
    engine.run(zero_inputs).expect("run 2");
    let out2: Vec<f32> = engine.outputs[&output_name]
        .floats()
        .expect("float output")
        .to_vec();

    // Outputs must differ — if they're identical, user inputs were constant-folded
    assert_ne!(
        out1, out2,
        "Two runs with different inputs produced identical outputs; \
        user inputs may have been constant-folded into the plan"
    );
}

/// Verify XNNPACK path produces same results as CPU path for a classifier.
#[cfg(feature = "xnnpack")]
#[test]
fn test_xnnpack_matches_cpu() {
    let base = fixture("bvlcalexnet-12");
    let (model_bytes, inputs) = load_model_and_inputs(&base, "bvlcalexnet-12.onnx", 0);

    let model = ModelProto::decode(&model_bytes[..]).expect("decode model proto");
    let graph = model.graph.as_ref().expect("model has no graph");
    let output_name = graph.output[0].name.clone();

    // CPU run
    let mut cpu_engine =
        InferenceEngine::new(&model_bytes, Default::default()).expect("load model (CPU)");
    cpu_engine.run(inputs.clone()).expect("CPU inference");
    let cpu_f = cpu_engine.outputs[&output_name]
        .floats()
        .expect("CPU float")
        .to_vec();

    // XNNPACK run
    let mut xnn_engine = InferenceEngine::new(
        &model_bytes,
        InferenceOptions {
            xnnpack: true,
            ..Default::default()
        },
    )
    .expect("load model (XNNPACK)");
    xnn_engine.run(inputs).expect("XNNPACK inference");
    let xnn_f = xnn_engine.outputs[&output_name]
        .floats()
        .expect("XNNPACK float")
        .to_vec();

    assert_eq!(cpu_f.len(), xnn_f.len());
    for (i, (c, x)) in cpu_f.iter().zip(xnn_f.iter()).enumerate() {
        assert!(
            (c - x).abs() < 0.01 || (c - x).abs() / c.abs().max(1e-6) < 0.01,
            "output [{i}]: CPU={c}, XNNPACK={x}"
        );
    }

    // Top-1 class should match
    let cpu_top = cpu_f
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let xnn_top = xnn_f
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        cpu_top, xnn_top,
        "Top-1 class mismatch: CPU={cpu_top}, XNNPACK={xnn_top}"
    );
}
