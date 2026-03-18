use std::collections::HashMap;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use onnx_infer::Tensor;
use onnx_infer::dims;

fn make_values(pairs: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn rand_vec(n: usize) -> Vec<f32> {
    // deterministic pseudo-random for reproducibility
    let mut v = vec![0.0f32; n];
    let mut state: u32 = 0xDEAD_BEEF;
    for x in v.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *x = (state as f32) / (u32::MAX as f32) * 2.0 - 1.0;
    }
    v
}

// --- Conv benchmarks ---

fn bench_conv_3x3_64(c: &mut Criterion) {
    use onnx_infer::layers::Layer;
    use onnx_infer::layers::conv::Conv;

    // Typical: batch=1, 64 in channels, 64 out channels, 56x56, 3x3 kernel
    let n = 1;
    let c_in = 64;
    let c_out = 64;
    let h = 56;
    let w = 56;
    let kh = 3;
    let kw = 3;

    let input = Tensor::new(dims![n, c_in, h, w], rand_vec(n * c_in * h * w));
    let weight = Tensor::new(dims![c_out, c_in, kh, kw], rand_vec(c_out * c_in * kh * kw));
    let bias = Tensor::new(dims![c_out], rand_vec(c_out));

    let values = make_values(vec![("x", input), ("w", weight), ("b", bias)]);
    let mut output = Tensor::default();

    let mut conv = Conv::new(
        vec!["x".into(), "w".into(), "b".into()],
        vec![kh as i64, kw as i64],
        vec![1, 1],
        vec![1, 1, 1, 1], // padding=1 for same output size
        vec![1, 1],
        1,
        String::new(),
        &[n, c_in, h, w],
        &[c_out, c_in, kh, kw],
    );

    c.bench_function("conv_3x3_c64_56x56", |b| {
        b.iter(|| conv.execute(&values, &mut output).unwrap())
    });
}

fn bench_conv_1x1_128(c: &mut Criterion) {
    use onnx_infer::layers::Layer;
    use onnx_infer::layers::conv::Conv;

    let n = 1;
    let c_in = 128;
    let c_out = 128;
    let h = 28;
    let w = 28;
    let kh = 1;
    let kw = 1;

    let input = Tensor::new(dims![n, c_in, h, w], rand_vec(n * c_in * h * w));
    let weight = Tensor::new(dims![c_out, c_in, kh, kw], rand_vec(c_out * c_in * kh * kw));
    let bias = Tensor::new(dims![c_out], rand_vec(c_out));

    let values = make_values(vec![("x", input), ("w", weight), ("b", bias)]);
    let mut output = Tensor::default();

    let mut conv = Conv::new(
        vec!["x".into(), "w".into(), "b".into()],
        vec![kh as i64, kw as i64],
        vec![1, 1],
        vec![0, 0, 0, 0],
        vec![1, 1],
        1,
        String::new(),
        &[n, c_in, h, w],
        &[c_out, c_in, kh, kw],
    );

    c.bench_function("conv_1x1_c128_28x28", |b| {
        b.iter(|| conv.execute(&values, &mut output).unwrap())
    });
}

// --- MatMul benchmarks ---

fn bench_matmul_256(c: &mut Criterion) {
    use onnx_infer::layers::Layer;
    use onnx_infer::layers::matmul::MatMul;

    let m = 256;
    let k = 256;
    let n = 256;

    let a = Tensor::new(dims![m, k], rand_vec(m * k));
    let b = Tensor::new(dims![k, n], rand_vec(k * n));

    let values = make_values(vec![("a", a), ("b", b)]);
    let mut output = Tensor::default();

    let mut matmul = MatMul::new(vec!["a".into(), "b".into()], &[m, k], &[k, n]);

    c.bench_function("matmul_256x256x256", |b| {
        b.iter(|| matmul.execute(&values, &mut output).unwrap())
    });
}

fn bench_matmul_512(c: &mut Criterion) {
    use onnx_infer::layers::Layer;
    use onnx_infer::layers::matmul::MatMul;

    let m = 512;
    let k = 512;
    let n = 512;

    let a = Tensor::new(dims![m, k], rand_vec(m * k));
    let b = Tensor::new(dims![k, n], rand_vec(k * n));

    let values = make_values(vec![("a", a), ("b", b)]);
    let mut output = Tensor::default();

    let mut matmul = MatMul::new(vec!["a".into(), "b".into()], &[m, k], &[k, n]);

    c.bench_function("matmul_512x512x512", |b| {
        b.iter(|| matmul.execute(&values, &mut output).unwrap())
    });
}

fn bench_matmul_batch(c: &mut Criterion) {
    use onnx_infer::layers::Layer;
    use onnx_infer::layers::matmul::MatMul;

    let batch = 8;
    let m = 64;
    let k = 64;
    let n = 64;

    let a = Tensor::new(dims![batch, m, k], rand_vec(batch * m * k));
    let b = Tensor::new(dims![batch, k, n], rand_vec(batch * k * n));

    let values = make_values(vec![("a", a), ("b", b)]);
    let mut output = Tensor::default();

    let mut matmul = MatMul::new(vec!["a".into(), "b".into()], &[batch, m, k], &[batch, k, n]);

    c.bench_function("matmul_8x64x64x64", |b| {
        b.iter(|| matmul.execute(&values, &mut output).unwrap())
    });
}

// --- Gemm benchmarks ---

fn bench_gemm_256(c: &mut Criterion) {
    use onnx_infer::layers::Layer;
    use onnx_infer::layers::gemm::Gemm;

    let m = 256;
    let k = 256;
    let n = 256;

    let a = Tensor::new(dims![m, k], rand_vec(m * k));
    let b = Tensor::new(dims![k, n], rand_vec(k * n));
    let bias = Tensor::new(dims![n], rand_vec(n));

    let values = make_values(vec![("a", a), ("b", b), ("c", bias)]);
    let mut output = Tensor::default();

    let mut gemm = Gemm::new(
        vec!["a".into(), "b".into(), "c".into()],
        1.0,
        1.0,
        false,
        false,
        &[m, k],
        &[k, n],
    );

    c.bench_function("gemm_256x256x256", |b| {
        b.iter(|| gemm.execute(&values, &mut output).unwrap())
    });
}

criterion_group!(
    benches,
    bench_conv_3x3_64,
    bench_conv_1x1_128,
    bench_matmul_256,
    bench_matmul_512,
    bench_matmul_batch,
    bench_gemm_256,
);
criterion_main!(benches);
