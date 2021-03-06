import torch
import os 
import sys 
import pandas as pd

sys.path.append("../train")

from data_loader import ImageAndPathDataset, npy_loader
import evaluation
import torchvision
import torch.nn as nn
from vgg import _vgg


import logging
from basic_model import BasicNet
from resnet import ResNet, Bottleneck, BasicBlock
from vgg import _vgg
from mnas_net import MNASNet
from densenet import DenseNet
from statistics import mean
import argparse
import time

def main(chosen_model, model_checkpoint, alternate_scaling=False):
    data = "/data2/cdcm/preprocessed/test" if not alternate_scaling else "/data2/cdcm/rescaled/preprocessed/test"
    test_dataset = ImageAndPathDataset(
        root = data,
        loader = npy_loader,
        extensions=".npy"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 128,
        # pin_memory=True
    )
    # device = torch.device("cpu")
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


    # model = _vgg("vgg16", "D", True)


    if "resnet50" in chosen_model.lower():
        model = ResNet(Bottleneck, [3,4,6,3], 4)
    elif chosen_model.lower() == "resnet34":
        model = ResNet(BasicBlock, [3,4,6,3], 4)
    elif "resnet18" in chosen_model.lower():
        model = ResNet(BasicBlock, [2,2,2,2], 4)
    elif chosen_model.lower() == "basic":
        model = BasicNet()
    elif "vgg_dropout" in chosen_model.lower():
        model = _vgg("vgg16", "D", True, dropout=True)
    elif "vgg" in chosen_model.lower():
        model = _vgg("vgg16", "D", True)
    elif chosen_model.lower() == "mnasnet":
        model = MNASNet(1.0)
    elif chosen_model.lower() == "densenet":
        model = DenseNet()
    else:
        print(f"Invalid option for model: {chosen_model}")
        exit(0)

    model = nn.DataParallel(model)
    weight_path = f"/data2/cdcm/models/{chosen_model}-{model_checkpoint}.pt" if not alternate_scaling else f"/data2/cdcm/models/rescaled/{chosen_model}-{model_checkpoint}.pt"
    checkpoint = torch.load(weight_path, map_location=device)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint["model_state_dict"].items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v

    model=model.to(device=device)
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch = next(iter(test_loader))
    times = []
    with torch.no_grad():
        for sample in batch[0]:
            # print(sample)
            start = time.time()
            sample = torch.unsqueeze(sample, 0)
            end = time.time()

            times.append(end - start)
    prediction_time = mean(times)
    with open(f"./results/{chosen_model}-{model_checkpoint}-prediction-time.txt", "w") as f:
        f.write(str(prediction_time))
        f.write("\n")
    # return

    evaluation.evaluate(model, test_loader, test_dataset.classes, device, "test", f"{chosen_model}-{model_checkpoint}")
    times = []



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
        help="name of model to evaluate, (resnet50, resnet34, resnet18, vgg, mnasnet, densenet)")
    parser.add_argument("checkpoint",
        help="model weight checkpoint to load")
    parser.add_argument("--alternate_scaling", "-a", required=False, action="store_true")
    args = parser.parse_args()

    chosen_model = args.model
    model_checkpoint = args.checkpoint
    main(chosen_model, model_checkpoint, alternate_scaling=args.alternate_scaling)