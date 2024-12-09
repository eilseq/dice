import argparse
import numpy as np
import onnxruntime as ort
import onnx
import os
import torch
import torch.onnx

from onnxsim import simplify
from rich.text import Text
from rich.console import Console

from dice_models.architectures import ConvAutoencoder, AttentionUNet, DiceArchitecture

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Load a DICE model")
    parser.add_argument('--id', type=str, required=True,
                        help='Model identifier')
    parser.add_argument('--preset', type=str,
                        required=True, help='Pattern preset')
    parser.add_argument('--architecture', type=str,
                        required=True, help='Model architecture')
    parser.add_argument('--loss', type=str, required=True,
                        help='Loss function')
    return parser.parse_args()


def simplify_onnx(model_onnx):
    model_simp, check = simplify(model_onnx)

    assert check, "Simplified ONNX model could not be validated"

    # Find the Resize node and modify it to assure backwards compatibility
    for node in model_simp.graph.node:
        if node.op_type == 'Resize':
            for attr in node.attribute:
                if attr.name == 'mode' and attr.s.decode('utf-8') == 'linear':
                    # Change interpolation mode to 'nearest'
                    attr.s = 'nearest'.encode('utf-8')

    return model_simp


def inference_onnx(model_onnx, input_tensor):
    session = ort.InferenceSession(model_onnx)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})[0]
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    return outputs


def inference_torch(model_tocrh, input_tensor):
    model_tocrh.eval()
    with torch.no_grad():
        outputs = model_tocrh(input_tensor)
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
    return outputs


def get_workspace_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))


def get_dist_path():
    return os.path.join(get_workspace_path(), "dist", "models")


def loadDiceModel(model_torch_path, architecture: DiceArchitecture):
    state_dict = torch.load(model_torch_path, weights_only=False)
    match DiceArchitecture(architecture):
        case DiceArchitecture.CONVOLUTIONAL_AUTO_ENCODER:
            model = ConvAutoencoder()
        case DiceArchitecture.ATTENTION_UNET:
            model = AttentionUNet(num_channels=1)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    console.print("[bold cyan]DICE Model Conversion - ONNX")
    args = parse_args()
    dummy_input = torch.randn(1, 1, 16, 16)

    model_filename = f"{args.preset}-{args.architecture}-{args.loss}"
    model_torch_path = os.path.join(get_dist_path(), model_filename + ".pth")
    model_onnx_path = os.path.join(get_dist_path(), model_filename + ".onnx")

    model_torch = loadDiceModel(model_torch_path, args.architecture)
    model_torch.eval()

    torch.onnx.export(
        model_torch,
        dummy_input,
        model_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    model_onnx = onnx.load(model_onnx_path)
    model_onnx = simplify_onnx(model_onnx)

    # Perform inference on both PyTorch and ONNX models
    outputs_torch = inference_torch(model_torch, dummy_input)
    outputs_onnx = inference_onnx(model_onnx_path, dummy_input.numpy())

    # Verify outputs are identical
    assert np.array_equal(outputs_torch.numpy(),
                          outputs_onnx), "Results are not similar"

    onnx.save(model_onnx, model_onnx_path)
    console.print("[bold green]Model successfully converted")
    console.print(Text(f"Saved at destination {
                  model_onnx_path}", style="yellow"))
