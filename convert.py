import onnx
import numpy as np
import torch
from PIL import Image
model = torch.load('./model.pt').to(torch.device('cuda'))
ONNX_PATH="./my_model.onnx"
model.eval()

data = np.asarray(Image.open('./results/LR/MyImage/chip.png'))
data = np.expand_dims(np.ascontiguousarray(data.transpose((2, 0, 1))), axis=0)
# lr = torch.tensor(data).float()



torch.onnx.export(
    model=model.module,
    args=torch.from_numpy(data).float().to(torch.device('cuda')), 
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=True,  # fold constant values for optimization
    input_names=['input'],
    output_names=['output']
)
# onnx_model = onnx.load(ONNX_PATH)
# onnx.checker.check_model(onnx_model)
