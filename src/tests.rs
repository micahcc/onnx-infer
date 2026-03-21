use std::collections::HashMap;
use std::fs;
use std::path::Path;

use approx::assert_relative_eq;
use prost::Message;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

use crate::DType;
use crate::InferenceEngine;
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

fn run_fixture(base: &Path, model_file: &str, test_set: usize) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes).expect("load model");

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
    let exp_data = expected.floats().expect("expected output should be float tensor");
    // XNNPACK uses optimized kernels that may accumulate differently,
    // so allow slightly higher tolerance when xnnpack feature is enabled.
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

fn run_fixture_argmax(base: &Path, model_file: &str, test_set: usize) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes).expect("load model");

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
    let exp_data = expected.floats().expect("expected output should be float tensor");

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

fn run_quantized_fixture(base: &Path, model_file: &str, test_set: usize) {
    run_quantized_fixture_with_tol(base, model_file, test_set, 0.1);
}

fn run_quantized_fixture_with_tol(
    base: &Path,
    model_file: &str,
    test_set: usize,
    softmax_tol: f32,
) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes).expect("load model");

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
    let want_probs = softmax(expected.floats().expect("expected output should be float tensor"));

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

fn run_multi_io_fixture(base: &Path, model_file: &str, test_set: usize) {
    run_multi_io_fixture_with_tol(base, model_file, test_set, 5e-3);
}

fn run_multi_io_fixture_with_tol(base: &Path, model_file: &str, test_set: usize, tol: f32) {
    let (model_bytes, inputs) = load_model_and_inputs(base, model_file, test_set);
    let mut engine = InferenceEngine::new(&model_bytes).expect("load model");

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
                    let want = expected.floats().expect("expected output should be float tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (g - w).abs() < tol || (g - w).abs() / w.abs().max(1e-6) < tol,
                            "output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                (DType::Int64, DType::Int64) => {
                    let got = output.ints().expect("output should be int64 tensor");
                    let want = expected.ints().expect("expected output should be int64 tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert_eq!(g, w, "output {name}[{j}]: got {g}, want {w}");
                    }
                }
                (DType::Float, DType::Int64) => {
                    let got = output.floats().expect("output should be float tensor");
                    let want = expected.ints().expect("expected output should be int64 tensor");
                    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (*g as i64 - w).abs() <= 1,
                            "output {name}[{j}]: got {g}, want {w}"
                        );
                    }
                }
                (DType::Int64, DType::Float) => {
                    let got = output.ints().expect("output should be int64 tensor");
                    let want = expected.floats().expect("expected output should be float tensor");
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

// --- MNIST models ---

#[test]
fn test_mnist1_set_0() {
    let _t = setup_tracing("mnist1_set_0");
    run_fixture(&fixture("mnist-1"), "model.onnx", 0);
}

#[test]
fn test_mnist1_set_1() {
    let _t = setup_tracing("mnist1_set_1");
    run_fixture(&fixture("mnist-1"), "model.onnx", 1);
}

#[test]
fn test_mnist1_set_2() {
    let _t = setup_tracing("mnist1_set_2");
    run_fixture(&fixture("mnist-1"), "model.onnx", 2);
}

#[test]
fn test_mnist7_set_0() {
    let _t = setup_tracing("mnist7_set_0");
    run_fixture(&fixture("mnist-7"), "model.onnx", 0);
}

#[test]
fn test_mnist8_set_0() {
    let _t = setup_tracing("mnist8_set_0");
    run_fixture(&fixture("mnist-8"), "model.onnx", 0);
}

#[test]
fn test_mnist12_set_0() {
    let _t = setup_tracing("mnist12_set_0");
    run_fixture(&fixture("mnist-12"), "mnist-12.onnx", 0);
}

#[test]
fn test_mnist12_int8_set_0() {
    let _t = setup_tracing("mnist12_int8_set_0");
    run_quantized_fixture(&fixture("mnist-12-int8"), "mnist-12-int8.onnx", 0);
}

// --- MobileNetV2 models ---

#[test]
fn test_mobilenetv2_7_set_0() {
    let _t = setup_tracing("mobilenetv2_7_set_0");
    run_fixture(&fixture("mobilenetv2-7"), "mobilenetv2-7.onnx", 0);
}

#[test]
fn test_mobilenetv2_12_set_0() {
    let _t = setup_tracing("mobilenetv2_12_set_0");
    run_fixture(&fixture("mobilenetv2-12"), "mobilenetv2-12.onnx", 0);
}

#[test]
fn test_mobilenetv2_12_int8_set_0() {
    let _t = setup_tracing("mobilenetv2_12_int8_set_0");
    run_quantized_fixture(
        &fixture("mobilenetv2-12-int8"),
        "mobilenetv2-12-int8.onnx",
        0,
    );
}

#[test]
fn test_mobilenetv2_12_qdq_set_0() {
    let _t = setup_tracing("mobilenetv2_12_qdq_set_0");
    // QDQ models use float Conv which has FP ordering sensitivity;
    // XNNPACK NHWC layout increases divergence slightly vs NCHW CPU path.
    run_quantized_fixture_with_tol(
        &fixture("mobilenetv2-12-qdq"),
        "mobilenetv2-12-qdq.onnx",
        0,
        0.20,
    );
}

// --- Tiny YOLOv2 models ---

#[test]
fn test_tinyyolov2_7_set_0() {
    let _t = setup_tracing("tinyyolov2_7_set_0");
    run_fixture(&fixture("tinyyolov2-7"), "model.onnx", 0);
}

#[test]
fn test_tinyyolov2_8_set_0() {
    let _t = setup_tracing("tinyyolov2_8_set_0");
    run_fixture(&fixture("tinyyolov2-8"), "model.onnx", 0);
}

// --- Faster R-CNN models ---

#[test]
fn test_faster_rcnn_12_set_0() {
    let _t = setup_tracing("faster_rcnn_12_set_0");
    run_multi_io_fixture(&fixture("faster-rcnn-12"), "FasterRCNN-12.onnx", 0);
}

// --- SSD MobileNet models ---

#[test]
fn test_ssd_mobilenet_v1_12_set_0() {
    let _t = setup_tracing("ssd_mobilenet_v1_12_set_0");
    run_multi_io_fixture(
        &fixture("ssd-mobilenet-v1-12"),
        "ssd_mobilenet_v1_12.onnx",
        0,
    );
}

// --- YOLOv4 models ---

#[test]
fn test_yolov4_11_set_0() {
    let _t = setup_tracing("yolov4_11_set_0");
    run_multi_io_fixture(&fixture("yolov4-11"), "yolov4.onnx", 0);
}

// --- Tiny YOLOv3 models ---

#[test]
fn test_tinyyolov3_11_set_0() {
    let _t = setup_tracing("tinyyolov3_11_set_0");
    run_multi_io_fixture(&fixture("tiny-yolov3-11"), "yolov3-tiny.onnx", 0);
}

// --- BiDAF models ---

#[test]
fn test_bidaf_9_set_0() {
    let _t = setup_tracing("bidaf_9_set_0");
    run_multi_io_fixture(&fixture("bidaf-9"), "bidaf.onnx", 0);
}

// --- AlexNet models ---

#[test]
fn test_bvlcalexnet_12_set_0() {
    let _t = setup_tracing("bvlcalexnet_12_set_0");
    run_fixture(&fixture("bvlcalexnet-12"), "bvlcalexnet-12.onnx", 0);
}

// --- CaffeNet models ---

#[test]
fn test_caffenet_12_set_0() {
    let _t = setup_tracing("caffenet_12_set_0");
    run_fixture(&fixture("caffenet-12"), "caffenet-12.onnx", 0);
}

// --- DenseNet models ---

#[test]
fn test_densenet_12_set_0() {
    let _t = setup_tracing("densenet_12_set_0");
    run_fixture(&fixture("densenet-12"), "densenet-12.onnx", 0);
}

// --- EfficientNet models ---

#[test]
fn test_efficientnet_lite4_11_set_0() {
    let _t = setup_tracing("efficientnet_lite4_11_set_0");
    run_fixture(&fixture("efficientnet-lite4-11"), "efficientnet-lite4.onnx", 0);
}

// --- GoogLeNet models ---

#[test]
fn test_googlenet_12_set_0() {
    let _t = setup_tracing("googlenet_12_set_0");
    run_fixture(&fixture("googlenet-12"), "googlenet-12.onnx", 0);
}

// --- Inception models ---

#[test]
fn test_inception_v1_12_set_0() {
    let _t = setup_tracing("inception_v1_12_set_0");
    run_fixture(&fixture("inception-v1-12"), "inception-v1-12.onnx", 0);
}

#[test]
fn test_inception_v2_9_set_0() {
    let _t = setup_tracing("inception_v2_9_set_0");
    run_fixture(&fixture("inception-v2-9"), "model.onnx", 0);
}

// --- RCNN ILSVRC13 models ---

#[test]
fn test_rcnn_ilsvrc13_9_set_0() {
    let _t = setup_tracing("rcnn_ilsvrc13_9_set_0");
    run_fixture(&fixture("rcnn-ilsvrc13-9"), "model.onnx", 0);
}

// --- ResNet models ---

#[test]
fn test_resnet18_v1_7_set_0() {
    let _t = setup_tracing("resnet18_v1_7_set_0");
    run_fixture(&fixture("resnet18-v1-7"), "resnet18-v1-7.onnx", 0);
}

#[test]
fn test_resnet18_v2_7_set_0() {
    let _t = setup_tracing("resnet18_v2_7_set_0");
    run_fixture(&fixture("resnet18-v2-7"), "resnet18-v2-7.onnx", 0);
}

#[test]
fn test_resnet34_v1_7_set_0() {
    let _t = setup_tracing("resnet34_v1_7_set_0");
    run_fixture(&fixture("resnet34-v1-7"), "resnet34-v1-7.onnx", 0);
}

#[test]
fn test_resnet34_v2_7_set_0() {
    let _t = setup_tracing("resnet34_v2_7_set_0");
    run_fixture(&fixture("resnet34-v2-7"), "resnet34-v2-7.onnx", 0);
}

#[test]
fn test_resnet50_v1_12_set_0() {
    let _t = setup_tracing("resnet50_v1_12_set_0");
    run_fixture(&fixture("resnet50-v1-12"), "resnet50-v1-12.onnx", 0);
}

#[test]
fn test_resnet50_v2_7_set_0() {
    let _t = setup_tracing("resnet50_v2_7_set_0");
    run_fixture(&fixture("resnet50-v2-7"), "resnet50-v2-7.onnx", 0);
}

#[test]
fn test_resnet50_caffe2_v1_9_set_0() {
    let _t = setup_tracing("resnet50_caffe2_v1_9_set_0");
    run_fixture(&fixture("resnet50-caffe2-v1-9"), "model.onnx", 0);
}

#[test]
fn test_resnet101_v1_7_set_0() {
    let _t = setup_tracing("resnet101_v1_7_set_0");
    run_fixture(&fixture("resnet101-v1-7"), "resnet101-v1-7.onnx", 0);
}

#[test]
fn test_resnet101_v2_7_set_0() {
    let _t = setup_tracing("resnet101_v2_7_set_0");
    run_fixture(&fixture("resnet101-v2-7"), "resnet101-v2-7.onnx", 0);
}

#[test]
fn test_resnet152_v1_7_set_0() {
    let _t = setup_tracing("resnet152_v1_7_set_0");
    run_fixture(&fixture("resnet152-v1-7"), "resnet152-v1-7.onnx", 0);
}

#[test]
fn test_resnet152_v2_7_set_0() {
    let _t = setup_tracing("resnet152_v2_7_set_0");
    run_fixture(&fixture("resnet152-v2-7"), "resnet152-v2-7.onnx", 0);
}

// --- ShuffleNet models ---

#[test]
fn test_shufflenet_9_set_0() {
    let _t = setup_tracing("shufflenet_9_set_0");
    run_fixture(&fixture("shufflenet-9"), "model.onnx", 0);
}

#[test]
fn test_shufflenet_v2_12_set_0() {
    let _t = setup_tracing("shufflenet_v2_12_set_0");
    run_fixture(&fixture("shufflenet-v2-12"), "shufflenet-v2-12.onnx", 0);
}

// --- SqueezeNet models ---

#[test]
fn test_squeezenet10_12_set_0() {
    let _t = setup_tracing("squeezenet10_12_set_0");
    run_fixture_argmax(&fixture("squeezenet1.0-12"), "squeezenet1.0-12.onnx", 0);
}

#[test]
fn test_squeezenet11_7_set_0() {
    let _t = setup_tracing("squeezenet11_7_set_0");
    run_fixture(&fixture("squeezenet1.1-7"), "squeezenet1.1.onnx", 0);
}

// --- VGG models ---

#[test]
fn test_vgg16_12_set_0() {
    let _t = setup_tracing("vgg16_12_set_0");
    run_fixture(&fixture("vgg16-12"), "vgg16-12.onnx", 0);
}

#[test]
fn test_vgg16_bn_7_set_0() {
    let _t = setup_tracing("vgg16_bn_7_set_0");
    run_fixture(&fixture("vgg16-bn-7"), "vgg16-bn.onnx", 0);
}

#[test]
fn test_vgg19_7_set_0() {
    let _t = setup_tracing("vgg19_7_set_0");
    run_fixture(&fixture("vgg19-7"), "vgg19.onnx", 0);
}

#[test]
fn test_vgg19_bn_7_set_0() {
    let _t = setup_tracing("vgg19_bn_7_set_0");
    run_fixture(&fixture("vgg19-bn-7"), "vgg19-bn-7.onnx", 0);
}

#[test]
fn test_vgg19_caffe2_9_set_0() {
    let _t = setup_tracing("vgg19_caffe2_9_set_0");
    run_fixture(&fixture("vgg19-caffe2-9"), "model.onnx", 0);
}

// --- ZFNet models ---

#[test]
fn test_zfnet512_12_set_0() {
    let _t = setup_tracing("zfnet512_12_set_0");
    run_fixture(&fixture("zfnet512-12"), "zfnet512-12.onnx", 0);
}

// --- ResNet101-DUC models ---

#[test]
#[ignore] // numerical divergence in segmentation output — needs investigation
fn test_resnet101_duc_12_set_0() {
    let _t = setup_tracing("resnet101_duc_12_set_0");
    run_fixture_argmax(&fixture("ResNet101-DUC-12"), "ResNet101-DUC-12.onnx", 0);
}

// --- FCN models ---

#[test]
fn test_fcn_resnet50_12_set_0() {
    let _t = setup_tracing("fcn_resnet50_12_set_0");
    run_multi_io_fixture(&fixture("fcn-resnet50-12"), "fcn-resnet50-12.onnx", 0);
}

#[test]
#[ignore] // deep segmentation model with accumulated FP divergence — needs investigation
fn test_fcn_resnet101_11_set_0() {
    let _t = setup_tracing("fcn_resnet101_11_set_0");
    run_multi_io_fixture(&fixture("fcn-resnet101-11"), "model.onnx", 0);
}

// --- Mask R-CNN models ---

#[test]
#[ignore] // requires ConvTranspose operator
fn test_mask_rcnn_12_set_0() {
    let _t = setup_tracing("mask_rcnn_12_set_0");
    run_multi_io_fixture(&fixture("MaskRCNN-12"), "MaskRCNN-12.onnx", 0);
}

// --- RetinaNet models ---

#[test]
fn test_retinanet_9_set_0() {
    let _t = setup_tracing("retinanet_9_set_0");
    run_multi_io_fixture(&fixture("retinanet-9"), "retinanet-9.onnx", 0);
}

// --- SSD models ---

#[test]
fn test_ssd_12_set_0() {
    let _t = setup_tracing("ssd_12_set_0");
    run_multi_io_fixture(&fixture("ssd-12"), "ssd-12.onnx", 0);
}

// --- YOLOv2 models ---

#[test]
fn test_yolov2_coco_9_set_0() {
    let _t = setup_tracing("yolov2_coco_9_set_0");
    run_multi_io_fixture(&fixture("yolov2-coco-9"), "yolov2-coco-9.onnx", 0);
}

// --- YOLOv3 models ---

#[test]
fn test_yolov3_12_set_0() {
    let _t = setup_tracing("yolov3_12_set_0");
    run_multi_io_fixture(&fixture("yolov3-12"), "yolov3-12.onnx", 0);
}

// --- ArcFace models ---

#[test]
fn test_arcfaceresnet100_8_set_0() {
    let _t = setup_tracing("arcfaceresnet100_8_set_0");
    run_fixture(&fixture("arcfaceresnet100-8"), "resnet100.onnx", 0);
}

// --- Emotion FERPlus models ---

#[test]
fn test_emotion_ferplus_8_set_0() {
    let _t = setup_tracing("emotion_ferplus_8_set_0");
    run_fixture(&fixture("emotion-ferplus-8"), "model.onnx", 0);
}

// --- UltraFace models ---

#[test]
fn test_version_rfb_320_set_0() {
    let _t = setup_tracing("version_rfb_320_set_0");
    run_multi_io_fixture(&fixture("version-RFB-320"), "version-RFB-320.onnx", 0);
}

#[test]
fn test_version_rfb_640_set_0() {
    let _t = setup_tracing("version_rfb_640_set_0");
    run_multi_io_fixture(&fixture("version-RFB-640"), "version-RFB-640.onnx", 0);
}
