use std::fs;
use std::path::PathBuf;

use clap::Parser;
use prost_reflect::DescriptorPool;
use prost_reflect::DynamicMessage;
use prost_reflect::SerializeOptions;

#[derive(Parser)]
#[command(name = "dump_onnx")]
#[command(about = "Dump ONNX model information as JSON", long_about = None)]
struct Args {
    /// Path to the .onnx file
    #[arg(value_name = "FILE")]
    input: PathBuf,
}

const DESCRIPTOR_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/onnx_descriptor.bin"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let bytes = fs::read(&args.input)?;

    let pool = DescriptorPool::decode(DESCRIPTOR_BYTES)?;
    let message_desc = pool
        .get_message_by_name("onnx.ModelProto")
        .expect("ModelProto descriptor");

    let dynamic = DynamicMessage::decode(message_desc, &bytes[..])?;

    let options = SerializeOptions::new()
        .skip_default_fields(true)
        .stringify_64_bit_integers(false)
        .use_proto_field_name(true);

    let mut serializer = serde_json::Serializer::pretty(std::io::stdout());
    dynamic.serialize_with_options(&mut serializer, &options)?;
    println!();

    Ok(())
}
