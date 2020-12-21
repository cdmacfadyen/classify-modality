"""
MVP of end to end training process, not used 
for final results. 
"""
import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import glob
import random
import pandas as pd
import numpy as np
import time
import argparse

import logging

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--machine", required = False, default = "")
args = parser.parse_args()

data_dir = args.data_dir
machine = args.machine

logging.basicConfig(filename=f'{machine}basic-net.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size = 3, padding= 1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8*56*56, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.pool2(out)
        out = out.view(-1, 8*56*56)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
logging.info(f"Training on device {device}.")
train_metadata = pd.read_csv(f"{data_dir}/preprocessed/train-metadata.csv")
train_metadata["path"] = train_metadata["path"].apply(lambda x : x[3:]) # remove first ../
 
train_metadata = train_metadata.sample(frac = 1, random_state = 4).reset_index(drop=True)
# train_metadata = train_metadata[:1000]
BATCH_SIZE = 1
labels = {
    "CT" : 0,
    "MR" : 1,
    "XR" : 2,
    "PT" : 3
}
def data_generator(metadata):
    i = 0
    limit = len(metadata)

    while True:
        i += 1
        if i == limit:
            i = 0
        x = np.load(metadata["path"][i])
        label = metadata["label"][i]
        encoding = labels[label]
        y = np.array([encoding])
        # print(f"Yielding {y}")
        yield x, y

gen = data_generator(train_metadata)
start = time.time()
def npy_loader(path):
    img = torch.from_numpy(np.load(path))
    img = img.unsqueeze(0)
    return img

def validate(model, train_loader, validate_loader, device): # loader adapted from deep learning in pytorch book
    # for name, loader in [("Train", train_loader), ("Validate", validate_loader)]:
    for name, loader in [("Validate", validate_loader)]:

        correct = 0
        total = 0

        with torch.no_grad():
            for samples, labels in loader:
                samples = samples.to(device=device)
                labels = labels.to(device=device)
                outputs = model(samples)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print(f"{name} accuracy: {(correct/total):.2f}")
        logging.info(f"{name} accuracy: {(correct/total):.2f}")

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

BATCH_SIZE = 64
train_dataset = datasets.DatasetFolder(
    root = f"{data_dir}/preprocessed/train",
    loader=npy_loader,
    extensions=(".npy",)
)

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
    )

validate_dataset = datasets.DatasetFolder(
    root = f"{data_dir}/preprocessed/validate",
    loader=npy_loader,
    extensions=(".npy",)
)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)


model = BasicNet().to(device=device)
loss_func = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=1e-2)


for epoch in range(0, 20):
    loss_train = 0.0
    i = 0
    correct=0
    total=0
    for x, y in train_loader:
        print(f"Epoch {epoch}: sample: {i}", end="\r")
        i += BATCH_SIZE
        x = x.to(device=device)
        y = y.to(device=device)
        outputs = model(x)
        loss = loss_func(outputs, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_train += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total += y.shape[0]
        correct += int((predicted == y).sum())
    print(f"Epoch {epoch}, loss {loss_train / len(train_loader)}")
    logging.info(f"Epoch {epoch}, loss {loss_train / len(train_loader)}")
    logging.info(f"Train accuracy: {(correct/total):.2f}")
    print(f"Train accuracy: {(correct/total):.2f}")
    validate(model, train_loader, validate_loader, device)



with torch.no_grad():
    for x, y in validate_loader:
        x = x.to(device=device)
        y = y.to(device=device)
        outputs = model(x)
        _, predicted = torch.max(outputs, dim=1)
        print([validate_dataset.classes[element] for element in predicted])
        logging.info([validate_dataset.classes[element] for element in predicted])
        break
        