import torch
import torchvision.models as models
model = models.resnet18(pretrained=True)

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

input_names = [ "./models/model3_38.pt" ]
output_names = [ "./models/model.onnx" ]

torch.onnx.export(model,
                 dummy_input,
                 "resnet18.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 )

