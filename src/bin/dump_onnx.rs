use std::fs;
use std::path::PathBuf;

use clap::Parser;
use prost::Message;

#[derive(Parser)]
#[command(name = "dump_onnx")]
#[command(about = "Dump ONNX model information", long_about = None)]
struct Args {
    /// Path to the .onnx file
    #[arg(value_name = "FILE")]
    input: PathBuf,
}

fn clear_raw_data(model: &mut onnx_infer::onnx::ModelProto) {
    if let Some(graph) = &mut model.graph {
        for initializer in &mut graph.initializer {
            initializer.raw_data.clear();
        }
        for node in &mut graph.node {
            for attr in &mut node.attribute {
                if let Some(tensor) = &mut attr.t {
                    tensor.raw_data.clear();
                }
                for tensor in &mut attr.tensors {
                    tensor.raw_data.clear();
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let bytes = fs::read(&args.input)?;
    let mut model = onnx_infer::onnx::ModelProto::decode(&bytes[..])?;
    clear_raw_data(&mut model);
    println!("{:#?}", model);
    Ok(())
}
