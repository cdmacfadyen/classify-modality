"""
Preprocessing code for image data. 
Resizes and rescales data using MedPicPy and 
then saves it as individual npy files for use in the training process. 

Because some scans are 3D, the number of 2D 
images can't be known ahead of time so 
a function was added to MedPicPy that 
loads the images individually 
and finds the required length of the output 
array, then another function that intialises an array of 
that length and loads all of the images into it. This is 
why the command line args optionally do a full run, which 
finds the length of the array, or a part run, which 
loads the length of the array from a file. 
"""

import time
import sys
import logging

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pandas as pd

from MedPicPy import medpicpy

parser = argparse.ArgumentParser()
parser.add_argument("dataset",
    help="train, validate or test")
parser.add_argument("data_dir",
    help="directory to load data from")
parser.add_argument("--output_dir", 
    required=False,
    help="directory to store data")
parser.add_argument("--modality", required=False, default = "",
    help="for debugging, only use images of this modality")
parser.add_argument("--test", required=False, action="store_true",
    help="for debugging, don't save any results")
parser.add_argument("-f", "--full_run", required=False, action="store_true",
    help="find the number of images (must be done for first run)")
parser.add_argument("-m", "--med_cache", required=False,
    help="direcotry for medpicpy cache")
args = parser.parse_args()

dataset = args.dataset
data_dir = args.data_dir
modality = ""
if args.modality:
    modality = args.modality

logging.basicConfig(filename=f'preprocessing-{dataset}.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True    # was cluttering log files

medpicpy.config.suppress_errors = True
medpicpy.config.rescale = True

if args.med_cache:
    medpicpy.config.set_cache_location(args.med_cache)
logging.debug(medpicpy.config.cache_location)

paths = []
file_count = 0
"""
Find the path to every image file within a directory and 
all of its subdirectories.
"""
root_dir = ""
if args.modality:
    root_dir = os.path.join(data_dir, dataset, args.modality)
else:
    root_dir = os.path.join(data_dir, dataset)

for root, dirs, files in os.walk(root_dir):
   for name in files:
       file_count += 1
       if file_count % 10000 == 0:
           logging.debug(f"Loaded {file_count} paths")
           logging.debug(os.path.join(root, name))
       paths.append(os.path.join(root, name))

start = time.time()

total_series_length = 0
image_paths = []

if args.full_run:
    total_series_length, image_paths = medpicpy.get_length_of_all_series(paths, skip_rgb=True)
    if not args.test:
        with open("./total-series-length.txt", "w") as f:
            f.write(f"{total_series_length}")   # so we don't have to run this more than once ideally.
        with open("./image_paths.txt", "w") as f:
            for path in image_paths:
                f.write(f"{path}\n")
else:
    with open("./total-series-length.txt", "r") as f:
        total_series_length_string = f.readline()
        total_series_length = int(total_series_length_string)
    
    with open("./image_paths.txt") as f:
        for line in f:
            image_paths.append(line[:-1])   # -1 to remove \n

images = medpicpy.load_all_slices_from_series(paths, total_series_length, (224, 224), use_memory_mapping=True)
# print(paths)

end = time.time()
logging.debug(f"Loaded images with shape: {images.shape}")
logging.debug(f"Took {end - start} seconds")
logging.debug(f"Loaded {images.nbytes // 1000000}MB")
selected_images = np.linspace(0, len(images), num=100, dtype=int, endpoint=False)
for index in selected_images:
    image = images[index]
    logging.debug(f"i: {index} MAX: {np.max(image)} MIN: {np.min(image)}")

fig, ax = plt.subplots()

ax.imshow(images[0], cmap="gray") # 10289 10288
plt.savefig(f"./images/test.png")

if args.test:
    print("Test run, exiting without updating files")
    medpicpy.clear_cache()
    exit(0)



for index in selected_images:
    image = images[index]
    logging.debug(f"i: {index} MAX: {np.max(image)} MIN: {np.min(image)}")

valid_modalities = ["MR", "CT", "PT", "CR", "DX", "MG"]
modalities = []

# Path is of form /unknown/dataset/modality/orig-dataset/maybe more/image
# This means modality is at least three from the end
# So this code starts there and looks backwards for something
# That matches exactly one of the valid modalities
for image_path in image_paths:
    print(image_path)
    path_tokens = Path(image_path).parts
    modality_index = -1
    while True:
        modality = path_tokens[modality_index]
        if any([modality == valid_modality for valid_modality in valid_modalities]):
            modalities.append(modality)
            break
        modality_index -= 1

labels = ["XR" if label == "CR" or label == "DX" or label == "MG" else label for label in modalities]

for index, image in enumerate(images):
    if args.output_dir:
        np.save(f"{args.output_dir}/preprocessed/{dataset}/{labels[index]}/{index}.npy", image, allow_pickle=False)
    else:
        np.save(f"{data_dir}/preprocessed/{dataset}/{labels[index]}/{index}.npy", image, allow_pickle=False)


metadata = pd.DataFrame()
saved_pathnames = [f"{dataset}/{labels[index]}/{index}.npy" for index in range(0, len(images))]
metadata["path"] = saved_pathnames
print(labels)
metadata["label"] = labels

datasets = [Path(image_path).parts[-2] for image_path in image_paths]
metadata["dataset"] = datasets

original_filenames = [Path(image_path).parts[-1] for image_path in image_paths]
metadata["original_filenames"] = original_filenames
if args.output_dir:
    metadata.to_csv(f"{args.output_dir}/preprocessed/{dataset}-metadata.csv", index=False)
else:
    metadata.to_csv(f"{data_dir}/preprocessed/{dataset}-metadata.csv", index=False)

# medpicpy.clear_cache()