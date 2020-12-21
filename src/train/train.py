"""
Code for training deep CNNs.
Takes the model name of the model to train 
and runs for the desired number of epochs.

Measures training and validation accuracy and 
stores this data to be graphed later. 

Implements adaptive learning rate. 
"""
import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import glob
import random
import pandas as pd
import numpy as np
import time
# from torch.utils.tensorboard import SummaryWriter
import json

import logging
from basic_model import BasicNet
from resnet import ResNet, Bottleneck, BasicBlock
from vgg import _vgg
from mnas_net import MNASNet
from densenet import DenseNet

from data_loader import NpyDataLoader, BatchedDataset
from evaluation import validate, print_batch_predictions, evaluate
import argparse
from simpleemailbot import EmailBot
import torch.utils.data as data

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("model")
parser.add_argument("checkpoint_dir")
parser.add_argument("--epochs", required=False, type=int)
parser.add_argument("--epochs_per_val", "-v", required=False, type=int)
parser.add_argument("--batched", required=False, action="store_true")
parser.add_argument("--weight_decay", "-w", required=False, type=float)

args = parser.parse_args()
data_dir = args.data_dir
chosen_model = args.model

logging.basicConfig(filename=f'train-{chosen_model}.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
logging.info(f"Training on device {device}.")

train_loader = None
validate_loader = None

classes = None
if args.batched:
    train_ds = BatchedDataset(f"{data_dir}/batched/train/")
    val_ds = BatchedDataset(f"{data_dir}/batched/validate/")
    train_loader = data.DataLoader(train_ds, 
        batch_size=None,
        shuffle=True,
        )
    validate_loader = data.DataLoader(
        val_ds,
        batch_size=None,
        shuffle=True,
        )
    classes = ["MR", "PT", "CT", "XR"]
else:
    loader = NpyDataLoader(data_dir, batch_size=128)
    train_loader = loader.train_loader
    validate_loader = loader.validate_loader
    classes = loader.classes
start = time.time()

model = None
weight_decay = 0.0
if args.weight_decay:
    weight_decay = args.weight_decay

if chosen_model.lower() == "resnet50":
    model = ResNet(Bottleneck, [3,4,6,3], 4)
elif chosen_model.lower() == "resnet34":
    model = ResNet(BasicBlock, [3,4,6,3], 4)
elif chosen_model.lower() == "resnet18":
    model = ResNet(BasicBlock, [2,2,2,2], 4)
elif chosen_model.lower() == "basic":
    model = BasicNet()
elif chosen_model.lower() == "vgg":
    model = _vgg("vgg16", "D", True)
elif chosen_model.lower() == "vgg_dropout":
    model = _vgg("vgg16", "D", True, dropout=True)
elif chosen_model.lower() == "mnasnet":
    model = MNASNet(1.0)
elif chosen_model.lower() == "densenet":
    model = DenseNet()
else:
    print(f"Invalid option for model: {chosen_model}")
    exit(0)
if args.weight_decay:
    chosen_model +=  f"w-{weight_decay}"
print(model)
model = model.to(device=device)
model = nn.DataParallel(model)  # distribute across multiple gpus for faster training

loss_func = nn.CrossEntropyLoss().to(device=device)
optimiser = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)

lr_scheduler = ReduceLROnPlateau(optimiser, "min", verbose=True, patience=3)    # adaptive learning rate
lr_scheduler_rate = len(train_loader) // 10
metrics = {
    "train_loss": [],
    "train_accuracy" : [],
    "validate_loss": [],
    "validate_accuracy": [],
    "train_epoch_time": []
    }
num_batches = len(train_loader)

num_epochs = 10
if args.epochs:
    num_epochs = args.epochs

epochs_per_validation = 2
if args.epochs_per_val:
    epochs_per_validation = args.epochs_per_val

last = time.time()
for epoch in range(0, num_epochs):
    lr_sched_loss = 0.0
    loss_train = 0.0
    i = 0
    correct = 0
    total = 0
    epoch_start = time.time()
    for x, y, path in train_loader:

        # batch_start = time.time()
        i += 1
        # to_dev_start = time.time()
        optimiser.zero_grad()
        x = x.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        # to_dev_end = time.time()
        # logging.debug(f"\tTo Device Time {to_dev_end-to_dev_start}")
        # pred_start = time.time()
        outputs = model(x)
        # pred_end = time.time()
        # logging.debug(f"\tPrediction Time {pred_end - pred_start}")
        loss = loss_func(outputs, y)
        # loss_start = time.time()
        loss.backward()
        # loss_end = time.time()
        # logging.debug(f"\tLoss backward pass {loss_end - loss_start}")
        # optim_start = time.time()
        optimiser.step()
        loss_train += loss.item()
        lr_sched_loss += loss.item()

        if i % lr_scheduler_rate == 0 and i != 0:
            lr_scheduler.step(lr_sched_loss / lr_scheduler_rate)
            logging.info(f"LR Sched Loss: {lr_sched_loss}")
            lr_sched_loss = 0.0
        
        _, predicted = torch.max(outputs, dim=1)
        total += y.shape[0]
        correct += int((predicted == y).sum())
        # optim_end = time.time()
        # logging.debug(f"\tOptim time {optim_end - optim_start}")
        batch_end = time.time()
        if i % 10 == 0:
            current = time.time()
            logging.debug(f"Batch time {(current - last) / 10}")
            print(f"Epoch {epoch}: batch: {i} / {num_batches}")
            logging.debug(f"EPOCH {epoch}: BATCH: {i} / {num_batches}")
            last = current

    epoch_end = time.time()
    print(f"Epoch {epoch}, loss {loss_train / len(train_loader)}")
    logging.info(f"Epoch {epoch}, loss {loss_train / len(train_loader)}")
    logging.info(f"Train accuracy: {(correct/total):.2f}")
    print(f"Train accuracy: {(correct/total):.2f}")

    # if epoch % 5 == 0:
    if epoch % epochs_per_validation == 0:
        val_accuracy, val_loss = validate(model, loss_func, validate_loader, device, epoch)
        metrics["validate_accuracy"].append(val_accuracy)
        metrics["validate_loss"].append(val_loss)
        checkpoint_path = f"{args.checkpoint_dir}/{chosen_model}-{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            "model_name": chosen_model,
            }, checkpoint_path)

    metrics["train_loss"].append(loss_train / len(train_loader))
    metrics["train_accuracy"].append(correct / total)
    metrics["train_epoch_time"].append(epoch_end - epoch_start)

print_batch_predictions(model, validate_loader, classes, device)
evaluate(model, validate_loader, classes, device, "validate", model_name=chosen_model)
with open(f"./results/{chosen_model}-training-metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

emailbot = EmailBot("cdcm@st-andrews.ac.uk", "update")
message = f"{chosen_model} finished training."
emailbot.email_me(message = message)