use maplit::hashmap;
use std::borrow::Cow;
use wonnx::{utils::InputTensor, utils::OutputTensor, Session};

const ONNX_MODEL: &[u8] = include_bytes!(env!("ONNX_MODEL_PATH"));

/// A provider for executing ONNX models.
pub struct OnnxProvider {
    session: Option<Session>,
}

impl OnnxProvider {
    /// Creates a new `OnnxProvider` with an uninitialized session.
    pub fn new() -> Self {
        Self { session: None }
    }

    /// Initializes the ONNX session asynchronously.
    ///
    /// # Errors
    /// Returns an error if the ONNX model cannot be loaded into a session.
    pub async fn init(&mut self) -> Result<(), String> {
        Session::from_bytes(ONNX_MODEL)
            .await
            .map(|session| {
                self.session = Some(session);
            })
            .map_err(|e| format!("Failed to create ONNX session: {}", e))
    }

    /// Runs the ONNX model with the provided input and processes the output.
    ///
    /// # Arguments
    /// * `input_flat` - A flat vector of `f32` values representing the input data.
    /// * `process_output` - A function that processes the output tensor.
    ///
    /// # Returns
    /// The processed output of type `R`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The ONNX session fails to execute.
    /// - The output tensor is missing or of an unexpected type.
    pub async fn run<F, R>(&self, input_flat: Vec<f32>, process_output: F) -> Result<R, String>
    where
        F: FnOnce(Vec<f32>) -> R,
    {
        // Ensure the session is initialized.
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "ONNX session is not initialized.".to_string())?;

        // Prepare the input tensor.
        let input_tensor = InputTensor::F32(Cow::Owned(input_flat));
        let input_map = hashmap! { "input".to_string() => input_tensor };

        // Execute the ONNX session.
        let output = session
            .run(&input_map)
            .await
            .map_err(|e| format!("Failed to run ONNX session: {}", e))?;

        // Extract and process the output tensor.
        let output_tensor = output
            .get("output")
            .ok_or_else(|| "Missing output tensor".to_string())?;

        match output_tensor {
            OutputTensor::F32(tensor) => {
                let processed_output = process_output(tensor.clone());
                Ok(processed_output)
            }
            _ => Err("Unexpected output tensor type".to_string()),
        }
    }
}
