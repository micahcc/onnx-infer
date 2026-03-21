use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(name = "onnx-infer")]
#[command(about = "Run ONNX model inference")]
struct Args {
    /// Path to the .onnx model file
    #[arg(value_name = "MODEL")]
    model: PathBuf,

    /// Input file(s) in the form name=path (e.g. input=image.png)
    /// Supports .pb (TensorProto), .png, .jpg/.jpeg images.
    /// For images, the tensor is shaped as [1, C, H, W] with float32 values in [0, 255].
    #[arg(value_name = "NAME=INPUT")]
    inputs: Vec<String>,

    /// Override input shape for images (e.g. "1,1,28,28")
    #[arg(long)]
    shape: Option<String>,

    /// Convert image to grayscale
    #[arg(long)]
    grayscale: bool,
}

fn load_input(
    path: &PathBuf,
    shape_override: &Option<Vec<usize>>,
    grayscale: bool,
) -> Result<onnx_infer::Tensor, Box<dyn std::error::Error>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pb" => {
            let bytes = fs::read(path)?;
            Ok(onnx_infer::Tensor::from_proto_bytes(&bytes)?)
        }
        "png" | "jpg" | "jpeg" | "bmp" | "gif" | "tiff" | "webp" => {
            let img = image::open(path)?;
            let img = if grayscale {
                image::DynamicImage::ImageLuma8(img.to_luma8())
            } else {
                img
            };

            let (w, h) = (img.width() as usize, img.height() as usize);
            let channels = if grayscale {
                1
            } else {
                img.color().channel_count() as usize
            };

            let (n, c, th, tw) = if let Some(shape) = shape_override {
                (shape[0], shape[1], shape[2], shape[3])
            } else {
                (1, channels, h, w)
            };

            // Resize if needed
            let img = if th != h || tw != w {
                img.resize_exact(tw as u32, th as u32, image::imageops::FilterType::Lanczos3)
            } else {
                img
            };

            // Convert to NCHW float32
            let raw = img.as_bytes();
            let mut data = vec![0.0f32; n * c * th * tw];
            for y in 0..th {
                for x in 0..tw {
                    for ch in 0..c {
                        let src_idx = (y * tw + x) * channels + ch;
                        let dst_idx = (ch * th + y) * tw + x;
                        data[dst_idx] = raw[src_idx] as f32;
                    }
                }
            }

            Ok(onnx_infer::Tensor::new(
                onnx_infer::dims![n, c, th, tw],
                data,
            ))
        }
        _ => Err(format!("Unsupported input format: {ext}").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let model_bytes = fs::read(&args.model)?;

    let shape_override: Option<Vec<usize>> = args.shape.as_ref().map(|s| {
        s.split(',')
            .map(|v| v.trim().parse::<usize>().expect("invalid shape dimension"))
            .collect()
    });

    let mut inputs = HashMap::new();
    for input_spec in &args.inputs {
        let (name, path) = input_spec
            .split_once('=')
            .ok_or_else(|| format!("Invalid input spec '{input_spec}', expected name=path"))?;
        let tensor = load_input(&PathBuf::from(path), &shape_override, args.grayscale)?;
        eprintln!("Input '{}': shape {:?}", name, tensor.dims);
        inputs.insert(name.to_string(), tensor);
    }

    let mut engine = onnx_infer::InferenceEngine::new(&model_bytes)?;
    engine.run(inputs)?;

    for (name, tensor) in &engine.outputs {
        println!("Output '{name}': shape {:?}", tensor.dims);
        match tensor.dtype() {
            onnx_infer::DType::Float => {
                let floats = tensor
                    .floats()
                    .expect("output tensor marked as Float but data is not f32");
                if floats.len() <= 100 {
                    println!("  values: {floats:?}");
                } else {
                    println!(
                        "  values: [{}, {}, ... ({} total)]",
                        floats[0],
                        floats[1],
                        floats.len()
                    );
                }
                if tensor.dims.len() == 2 && tensor.dims[0] == 1 {
                    let (class, score) = floats
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap();
                    println!("  argmax: class {class} (score {score:.4})");
                }
            }
            onnx_infer::DType::Int64 => {
                let ints = tensor
                    .ints()
                    .expect("output tensor marked as Int64 but data is not i64");
                if ints.len() <= 100 {
                    println!("  values: {ints:?}");
                } else {
                    println!(
                        "  values: [{}, {}, ... ({} total)]",
                        ints[0],
                        ints[1],
                        ints.len()
                    );
                }
            }
        }
    }

    Ok(())
}
