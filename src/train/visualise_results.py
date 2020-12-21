"""
Visualisation code. Takes the name 
of a model and plots the training and 
validation performance metrics, 
then saves them in the `./results/images` directory. 
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import itertools
from matplotlib import colorbar
import json

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("model")
parser.add_argument("title")
args = parser.parse_args()

plt.rcParams.update({"font.size":16})

dataset = args.dataset
model = args.model
title = args.title
actual_labels = np.load(f"./results/{model}-{dataset}-labels.npy")
predicted_labels = np.load(f"./results/{model}-{dataset}-predictions.npy")
class_probs = np.load(f"./results/{model}-{dataset}-class-probs.npy")
processed_image_paths = np.load(f"./results/{model}-{dataset}-paths.npy")
metrics = {}

with open(f"./results/{model}-training-metrics.json") as f:
    metrics = json.load(f)

fig, ax = plt.subplots()
x = range(len(metrics["train_loss"]))
x_factor = len(metrics["train_loss"]) // len(metrics["validate_loss"])
val_x = [num for num in x if num % x_factor == 0]
plt.subplots_adjust(top=0.85)
fig.suptitle(f"Training and Validation Set Loss per Epoch - {title}", wrap=True)
ax.plot(x, metrics["train_loss"], label="Train")
ax.plot(val_x, metrics["validate_loss"], label="Validate")
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.set_ylim(0)
ax.legend()
plt.savefig(f"./results/images/{model}-train-val-loss.png")

train_acc = [accuracy * 100 for accuracy in metrics["train_accuracy"]]
val_acc = [accuracy * 100 for accuracy in metrics["validate_accuracy"]]

fig, ax = plt.subplots()
plt.subplots_adjust(top=0.85)
pos = ax.get_position()
fig.suptitle(f"Training and Validation Set Accuracy per Epoch - {title}", wrap=True)
y = range(len(metrics["train_loss"]))
ax.plot(x, train_acc, label="Train")
ax.plot(val_x, val_acc, label="Validate")
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Epoch")
ax.set_ylim(0, 100)
ax.legend()
plt.savefig(f"./results/images/{model}-train-val-acc.png")

fig, ax = plt.subplots()
fig.suptitle(f"Time per Training Epoch - {title}")
x = range(len(metrics["train_epoch_time"]))
ax.plot(x, metrics["train_epoch_time"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Epoch Time (s)")
# ax.set_ylim(0)
plt.savefig(f"./results/images/{model}-epoch-time.png")

print(class_probs.shape)

## ACCURACY
# total_correct / total
with open(f"./results/images/{model}-report.txt", "w") as f:
    correct = actual_labels == predicted_labels
    num_correct = np.sum(correct)
    total = len(actual_labels)
    print(f"Total: {total}, Correct : {num_correct}, Accuracy: {num_correct / total:.2f}")
    f.write(f"Total: {total}, Correct : {num_correct}, Accuracy: {num_correct / total:.2f}\n")
    ## Balanced Accuracy
    # Defined by sklearn as the mean recall of all classes
    balanced_accuracy = balanced_accuracy_score(actual_labels, predicted_labels)
    print(f"Balanced Accuracy: {balanced_accuracy}")
    f.write(f"Balanced Accuracy: {balanced_accuracy}\n")

    ## CONFUSION MATRIX
    conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=[0,1,2,3])
    print(conf_matrix)
    f.write(str(conf_matrix))
    f.write("\n")
    classes = ""
    with open(f"./results/{model}-{dataset}-classes.csv") as class_file:
        line = class_file.readline()
        classes = line.split(",")
    classes = classes[:4]   # trailing comma in classes file
    fig, ax = plt.subplots(figsize=(10,10))
    img = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    cm = conf_matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        text = f"{cm[i,j]:0.0f}"
        plt.text(j, i, text, horizontalalignment='center', va='center', color=color, fontsize=25)
        ax.axhline(i-.5, color='black', linewidth=1.5)
        ax.axvline(j-.5, color='black', linewidth=1.5)

    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    ax.set_yticks(np.arange(4))
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    fig.suptitle(f"Confusion Matrix - {title}")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Number of images")
    plt.savefig(f"./results/images/{model}-{dataset}-confusion-matrix.png")

    ##4
    # Precision and recall
    precision, recall, _, _ = precision_recall_fscore_support(actual_labels, predicted_labels)
    for label  in classes:
        print(label, end="\t")
        f.write(f"{label}\t")
    print()
    f.write("\n")
    for arr in ( precision, recall):
        for i in range(len(classes)):
            print(f"{arr[i]:.2f}", end = "\t")
            f.write(f"{arr[i]:.2f}\t")
        print()
        f.write("\n")

