import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import itertools
from matplotlib import colorbar
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

def simplify_dataset_name(dataset_name):
    """In a couple of cases (BraTS and MURA) the dataset name is not quite correct
    because of a mistake made earlier in the pipeline. 
    This function transforms the dataset names into a more readable format.

    Args:
        dataset_name (string): name of dataset to simplify

    Returns:
        string: simplified dataset name
    """

    if "BraTS20" in dataset_name:
        return "BraTS20"
    elif "study" in dataset_name:
        return "MURA"
    else:
        return dataset_name
    
def visualise_model_results(dataset, model, title, trained_with_batch_order):

    actual_labels = np.load(f"./results/{model}-{dataset}-labels.npy")
    predicted_labels = np.load(f"./results/{model}-{dataset}-predictions.npy")
    class_probs = np.load(f"./results/{model}-{dataset}-class-probs.npy")
    processed_image_paths = np.load(f"./results/{model}-{dataset}-paths.npy")
    test_metadata = pd.read_csv("/data2/cdcm/preprocessed/test-metadata.csv")
    test_metadata = test_metadata.set_index("path").T.to_dict()

    """
    UNBATCHED (actual) CLASS ORDER: CT,MR,PT,XR,
    PREBATCHED CLASS ORDER:         MR,PT,CT,XR,

    The actual labels will always be unbatched order.
    So we want to transform the predicted labels to be in batched order. 
    To do that we make a new array with the same shape, 
    and switch the columns around. 
    """
    if trained_with_batch_order:
        new_predicted_labels = predicted_labels.copy()
        print(predicted_labels.shape, new_predicted_labels.shape)
        new_predicted_labels[predicted_labels == 0] = 1
        new_predicted_labels[predicted_labels == 1] = 2
        new_predicted_labels[predicted_labels == 2] = 0
        predicted_labels = new_predicted_labels
        # # new_predicted_labels[:,0] = predicted_labels[:,2]
        # # new_predicted_labels[:,1] = predicted_labels[:,0]
        # # new_predicted_labels[:,2] = predicted_labels[:,1]
        # # new_predicted_labels[:,3] = predicted_labels[:,3]
        # first_hundred = predicted_labels[:100]
        # # print(first_hundred)
        # # print(first_hundred == 0)
        # new_hundred = first_hundred.copy()
        # new_hundred[first_hundred == 0] = 1
        # new_hundred[first_hundred == 1] = 2
        # new_hundred[first_hundred == 2] = 0
        # print(first_hundred)
        # print(new_hundred)
        # first_hundred = new_hundred
        # print(first_hundred)
        # exit(0)
        # predicted_labels[predicted_labels == 0] = 1
        # predicted_labels[predicted_labels == 1] = 2
        # predicted_labels[predicted_labels == 2] = 0

    # for key in test_metadata:
    #     print(key)
    #     print(test_metadata[key])
    #     break
    # print(test_metadata)
    print(processed_image_paths)
    print(actual_labels)
    print(predicted_labels)
    datasets_correct = defaultdict(int)
    datasets_total = defaultdict(int)
    number_of_samples = len(actual_labels)
    with open(f"./results/images/{model}-report.txt", "w") as f:
        # how would one calculate the accuracy for every dataset?
        num_correct = 0
        for i in range(len(actual_labels)):
            print(f"Processed {i} of {number_of_samples}", end="\r")
            actual = actual_labels[i]
            predicted = predicted_labels[i]
            # image_path = Path(processed_image_paths[i])
            image_path = processed_image_paths[i]
            # print(actual, predicted)
            # print(actual == predicted)
            dataset_for_image = test_metadata[image_path]["dataset"]
            dataset_for_image = simplify_dataset_name(dataset_for_image)
            modality = test_metadata[image_path]["label"]
            if actual == predicted:
                num_correct += 1
                datasets_correct[f"{dataset_for_image} {modality}"] += 1
            datasets_total[f"{dataset_for_image} {modality}"] += 1
            
            # print(image_path)
            # print(test_metadata[image_path])
            # print(datasets_correct)
            # print(datasets_total)

        # correct = actual_labels == predicted_labels
        # num_correct = np.sum(correct)
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
        plt.rcParams.update({"font.size":30})

        fig, ax = plt.subplots(figsize=(12,10))
        img = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
        cm = conf_matrix
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            text = f"{cm[i,j]:0.0f}"
            plt.text(j, i, text, horizontalalignment='center', va='center', color=color, fontsize=25)
            ax.axhline(i-.5, color='black', linewidth=1.5)
            ax.axvline(j-.5, color='black', linewidth=1.5)

        # ax.tick_params(
        #     axis='both',
        #     which='both',
        #     labeltop=False,
        #     labelbottom=False,
        #     labelleft=False,
        #     labelright=False,
        #     length=0)
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")
        ax.set_yticks(np.arange(4))
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        fig.suptitle(f"Confusion Matrix - {title}")
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Number of Images")
        plt.savefig(f"./results/images/{model}-{dataset}-confusion-matrix.png")

        ##4
        # Precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(actual_labels, predicted_labels, labels=[0,1,2,3])
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
        datasets_percentage = defaultdict(float)
        for key in datasets_total:
            correct = datasets_correct[key]
            total = datasets_total[key]
            percent = correct / total
            datasets_percentage[key] = percent

        for dataset_for_image, percent in datasets_percentage.items():
            f.write(f"{dataset_for_image}, {percent}\n")
    # print(datasets_correct)
    # print(datasets_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("title")
    parser.add_argument("--trained_with_batch_order","-o", action="store_true", required=False)   # the class order that the model was trained on


    args = parser.parse_args()


    dataset = args.dataset
    model = args.model
    title = args.title
    order = args.trained_with_batch_order    #TODO: explain cock up
    visualise_model_results(dataset, model, title, order)






