import torch
from torch2trt import torch2trt
from models.experimental import attempt_load
#from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
#model = alexnet(pretrained=True).eval().cuda()
yolov5_file = r'moje_modele/best_nowe.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(yolov5_file, device=device)

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))