import torch
import torchvision
import logging
import os
import pandas as pd
from pathlib import Path
import numpy as np
logging.getLogger(__name__)

def validate(
    model,
    loss_func, 
    validate_loader, 
    device,
    epoch): # loader adapted from deep learning in pytorch book
    for name, loader in [("Validate", validate_loader)]:
        num_batches = len(loader)
        i = 0
        correct = 0
        total = 0
        with torch.no_grad():
            sum_loss = 0.0
            for samples, labels, path in loader:
                print(f"Validating: batch: {i} / {num_batches}" , end="\r")
                i += 1
                samples = samples.to(device=device)
                labels = labels.to(device=device)
                outputs = model(samples)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
                loss = loss_func(outputs, labels)
                sum_loss += loss.item()
        print(f"{name} accuracy: {(correct/total):.2f}")
        logging.info(f"{name} accuracy: {(correct/total):.2f}")
        # writer.add_scalar("Accuracy/validate", correct/total, epoch)
        # writer.add_scalar("Loss/validate",sum_loss / len(loader), epoch)
        accuracy = correct/total
        loss = sum_loss / len(loader)
        return accuracy, loss

def print_batch_predictions(model, loader, classes, device):
    """Print a batch of predictions as lables,
    not particularly informative but useful to ensure 
    the model does not only predict one class.

    Args:
        model (nn.Module): pytorch model to make predictions.
        loader (DataLoader): PyTorch dataloader for dataset.
        classes (list): List of classes
        device (): Pytorch device to compute on
    """
    with torch.no_grad():
        for x, y, path in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)
            print([classes[element] for element in predicted])
            logging.info([classes[element] for element in predicted])
            break

def evaluate(model, data_loader, classes, device, name, model_name = ""):
    """takes loader and saves all predictions, actual labels 
    and class probabilites for each sample

    Args:
        model ([type]): model to evaluate
        data_loader ([type]): data loader
        classes ([type]): list of classes from data loader
        device ([type]): device to run predictions on
        name ([type]): name of dataset (train, validate, test)
    """
    # clear results file
    # for every batch
    # write prediction, actual, prediction_probs,
    results_table = pd.DataFrame(columns=["Actual", "Predicted", "Scores"])
    with open(f"./results/{model_name}-{name}-classes.csv", "w") as f:
        for classname in classes:
            f.write(f"{classname},")
    class_probabilities = []
    predictions = []
    class_labels = []
    paths = []
    counter = 0
    batches = len(data_loader)
    with torch.no_grad():
        for x, y, path in data_loader:
            counter += 1
            print(f"Batch {counter} of {batches}", end="\r")
            x = x.to(device=device)
            y = y.to(device=device)
            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)

            class_probabilities.append(outputs)
            predictions.append(predicted)
            class_labels.append(y)
            paths.extend(["/".join(Path(full_path).parts[-3:]) for full_path in path])
            


    class_probs_tensor = torch.cat(class_probabilities) 
    preds_tensor = torch.cat(predictions)
    labels_tensor = torch.cat(class_labels)

    class_probs_np = class_probs_tensor.cpu().numpy()   
    preds_np = preds_tensor.cpu().numpy()   
    labels_np = labels_tensor.cpu().numpy()   

    np.save(f"./results/{model_name}-{name}-class-probs.npy", class_probs_np)
    np.save(f"./results/{model_name}-{name}-predictions.npy", preds_np)
    np.save(f"./results/{model_name}-{name}-labels.npy", labels_np)
    np.save(f"./results/{model_name}-{name}-paths.npy", paths)
# batch x classes
# class_probs batches x batch_size x n_classes
# samples x n_classes
