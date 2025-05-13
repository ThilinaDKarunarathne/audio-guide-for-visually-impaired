import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Load the SegFormer-B0 model from Hugging Face Transformers
model_name = "nvidia/segformer-b0-finetuned-cityscapes-768-768"
feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor matching the input size (Batch x Channels x Height x Width)
dummy_input = torch.randn(1, 3, 768, 768)  # 1 batch, 3 RGB channels, 640x480 resolution

# Export the model to ONNX
onnx_model_path = "segformer_b0_finetuned_cityscapes_768_768.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,  # Store trained parameters in the model file
    opset_version=11,    # ONNX version to export to
    input_names=["input"],  # Model's input name
    output_names=["output"],  # Model's output name
    dynamic_axes={  # Allow dynamic axes for inputs and outputs
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    }
)

print(f"Model has been successfully exported to {onnx_model_path}")