use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(name = "load_onnx")]
#[command(about = "Load and run ONNX model", long_about = None)]
struct Args {
    /// Path to the .onnx file
    #[arg(value_name = "FILE")]
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let bytes = fs::read(&args.input)?;
    let mut engine = onnx_infer::InferenceEngine::new(&bytes, HashMap::new())?;
    engine.run(HashMap::new())?;
    println!("{:#?}", engine.outputs);
    Ok(())
}
