import torch
import torchvision
import torch.nn as nn

import sys
sys.path.append("..")
from train.vgg import _vgg

model = _vgg("vgg16", "D", True, dropout=True)
model = nn.DataParallel(model) 
# checkpoint = torch.load("/data2/cdcm/models/vgg-4.pt")
checkpoint = torch.load("/scratch/other/cdcm/models/vgg_dropoutw-0.05-0.pt", map_location = torch.device("cpu"))

model.load_state_dict(checkpoint["model_state_dict"])
print(checkpoint["optimizer_state_dict"])
model.eval()
print(model)