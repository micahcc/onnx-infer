use std::env;
use std::fs;

use prost::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect_model <MODEL_PATH>");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let model_bytes = fs::read(model_path)?;
    let model = onnx_infer::onnx::ModelProto::decode(&*model_bytes)?;

    println!("Model IR version: {}", model.ir_version);
    println!("Producer name: {}", model.producer_name);
    println!("Producer version: {}", model.producer_version);

    let graph = model.graph.ok_or("No graph in model")?;
    println!("Graph name: {}", graph.name);

    println!("\nInputs:");
    for input in &graph.input {
        println!("  {}", input.name);
        if let Some(typ) = &input.r#type {
            if let Some(onnx_infer::onnx::type_proto::Value::TensorType(tt)) = &typ.value {
                if let Some(shape) = &tt.shape {
                    print!("    Shape: [");
                    let mut first = true;
                    for dim in &shape.dim {
                        if !first {
                            print!(", ");
                        }
                        first = false;
                        match &dim.value {
                            Some(
                                onnx_infer::onnx::tensor_shape_proto::dimension::Value::DimValue(v),
                            ) => {
                                print!("{v}");
                            }
                            Some(
                                onnx_infer::onnx::tensor_shape_proto::dimension::Value::DimParam(s),
                            ) => {
                                print!("{s}");
                            }
                            None => {
                                print!("?");
                            }
                        }
                    }
                    println!("]");
                }
                println!("    Data type: {}", tt.elem_type);
            }
        }
    }

    println!("\nOutputs:");
    for output in &graph.output {
        println!("  {}", output.name);
        if let Some(typ) = &output.r#type {
            if let Some(onnx_infer::onnx::type_proto::Value::TensorType(tt)) = &typ.value {
                if let Some(shape) = &tt.shape {
                    print!("    Shape: [");
                    let mut first = true;
                    for dim in &shape.dim {
                        if !first {
                            print!(", ");
                        }
                        first = false;
                        match &dim.value {
                            Some(
                                onnx_infer::onnx::tensor_shape_proto::dimension::Value::DimValue(v),
                            ) => {
                                print!("{v}");
                            }
                            Some(
                                onnx_infer::onnx::tensor_shape_proto::dimension::Value::DimParam(s),
                            ) => {
                                print!("{s}");
                            }
                            None => {
                                print!("?");
                            }
                        }
                    }
                    println!("]");
                }
                println!("    Data type: {}", tt.elem_type);
            }
        }
    }

    println!("\nOperators:");
    let mut op_counts = std::collections::HashMap::new();
    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }
    let mut op_types: Vec<_> = op_counts.keys().collect();
    op_types.sort();
    for op_type in op_types {
        println!("  {}: {}", op_type, op_counts[op_type]);
    }

    // Print some nodes with complex patterns
    println!("\nSample nodes:");
    let mut complex_nodes = Vec::new();
    for node in &graph.node {
        if node.op_type == "Upsample"
            || node.op_type == "Resize"
            || node.op_type == "NonMaxSuppression"
        {
            complex_nodes.push(node);
        }
    }

    for (i, node) in complex_nodes.iter().enumerate().take(10) {
        println!("Node {i}:");
        println!("  Op: {}", node.op_type);
        println!("  Name: {}", node.name);
        println!("  Inputs: {:?}", node.input);
        println!("  Outputs: {:?}", node.output);
        println!("  Attributes:");
        for attr in &node.attribute {
            println!("    {}: {:?}", attr.name, attr.r#type);
        }
    }

    Ok(())
}
