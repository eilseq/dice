use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Get the project path from the `CARGO_MANIFEST_DIR` environment variable
    let project_path =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by Cargo");

    // Default to "target/debug/default_model.onnx" if ONNX_MODEL_PATH is not set
    let model_path = env::var("ONNX_MODEL_PATH")
        .unwrap_or_else(|_| project_path + "/target/debug/default_model.onnx");

    // Create the dummy file if using the default path and it doesn't already exist
    if model_path == "target/debug/default_model.onnx" {
        let path = Path::new(&model_path);
        if !path.exists() {
            fs::write(path, "dummy model content").expect("Failed to create dummy ONNX model");
        }
    }

    // Pass the model path to the compiler environment
    println!("cargo:rerun-if-env-changed=ONNX_MODEL_PATH");
    println!("cargo:rustc-env=ONNX_MODEL_PATH={}", model_path);
}
