### Build Instructions for Dice Max External

This folder contains the source code for the **Dice** Max external, which enables ONNX-powered diffusion-based pattern generation. Before using the external, you must generate an ONNX model based on the instructions provided in the `models` folder.

---

## Building the Max External

### Prerequisites

1. **ONNX Model**:

   - You must build an ONNX model before proceeding. Follow the instructions in the `models` folder to train and export a model to be used by the Dice external.

2. **Rust Environment**:

   - Install Rust via [rustup.rs](https://rustup.rs/).

3. **Install Cargo-Make**:
   - Install [cargo-make](https://sagiegurari.github.io/cargo-make/) for building the project:
     ```bash
     cargo install cargo-make
     ```

---

### Build Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/dice.git
   cd dice/app
   ```

2. **Build Max External**:
   Use the path to the ONNX model to build the Max external:

   ```bash
   ONNX_MODEL_PATH="path_to_your_onnx_model.onnx" cargo make dice-external
   ```

3. **Install the External in Max**:
   Copy the built `dice` external to your Max `externals` folder or load it directly in your Max patch.

4. **QA Suite - Build MaxForLive Device**:
   Use the path to the ONNX model to build the MaxForLive Device with QA Suite (Ableton Live Set):

   ```bash
   ONNX_MODEL_PATH="path_to_your_onnx_model.onnx" cargo make dice-m4l
   ```

---

## Usage in Max

- **Input Data**:

  - Use the matrix UI to provide structured 16x16 matrix inputs.
  - Adjust parameters like `seed`, `noiseLevel`, and `threshold` interactively in Max to modify the generation.

- **Generate Patterns**:
  - Trigger pattern generation using the appropriate controls in your Max patch.
  - Combine generated patterns for creative experimentation.

---

For more details on the **DICE** Max external and the mono repository, refer to the `dice.maxhelp` file generated in the debug folder, next to the external artifact.
