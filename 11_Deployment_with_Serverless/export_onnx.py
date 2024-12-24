import torch
import onnx

def export_model(model_path, output_path):

    # Load the TorchScript model
    jit_model = torch.jit.load(model_path)

    # Set the model to evaluation mode
    jit_model.eval()

    # Define a dummy input with the required shape
    dummy_input = torch.randn(1, 3, 160, 160)

    # Export the model to ONNX
    onnx_file_path = output_path
    torch.onnx.export(
        jit_model,                       # The JIT model
        dummy_input,                     # Dummy input for the model
        onnx_file_path,                  # Output ONNX file path
        export_params=True,              # Store the trained weights in the model file
        do_constant_folding=True,        # Optimize the model by folding constants
        input_names=['input'],           # Name of the input tensor
        output_names=['output'],         # Name of the output tensor
        dynamic_axes={                   # Specify axes for dynamic sizes (optional)
            'input': {0: 'batch_size'},  # Dynamic batch size for input
            'output': {0: 'batch_size'}  # Dynamic batch size for output
        }
    )

    print(f"Model has been exported to {onnx_file_path}")

if __name__ == "__main__":
    export_model("model.pt", "model.onnx")