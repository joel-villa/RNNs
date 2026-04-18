import torch
import onnxruntime as ort

def save_model_onnx(model, save_path, device):
    model.eval()
    tensor_x = torch.rand((32, 50, 1), dtype=torch.float32).to(device)
    torch.onnx.export(model,                 # model to export
                  (tensor_x,),             # inputs of the model,
                  save_path,               # filename of the ONNX model
                  input_names=["input"],   # Rename inputs for the ONNX model
                  output_names=["output"], # Rename output
                  dynamo=True,             # True or False to select the exporter to use
                  dynamic_shapes={
                    "line_tensor": {0: torch.export.Dim("batch")}
                    }
                  )

def load_model_onnx(model_path, train_dataloader, test_dataloader):
    # load onnx model from path
    pass

def get_path(model_type, num_hidden_layers, num_hidden_neurons, learning_rate, num_epochs, ticker):
    return f"models/{model_type}_hl{num_hidden_layers}_hn{num_hidden_neurons}_lr{learning_rate}_e{num_epochs}_t{ticker}.onnx"