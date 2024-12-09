# DICE Model

This folder provides tools for generating and training machine learning models for pattern generation, using various architectures. It simplifies the process of creating, configuring, and testing models for experimentation and deployment using custom configuration files.

## Installation

### Clone the Repository:

```
git clone https://github.com/eilseq/dice.git
cd ./dice/models
```

### Create a Conda Environment:

```
conda create --name dice-models python=3.12
conda activate dice-models
```

### Install Dependencies:

```
pip install pytest torch onnx onnxruntime
pip install -e .
```

## Usage

### Run Tests

```
pytest
```

### Train a Model

```
python scripts/train_model.py --id test1 --preset default --architecture att_unet --loss mse_poly_penalty --noise_level 0.2
```

### Export to ONNX

```
python scripts/export_onnx.py --id test1 --preset default --architecture att_unet --loss mse_poly_penalty
```

## License

See [LICENSE](../LICENSE.md)
