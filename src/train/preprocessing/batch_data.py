"""Currently the training process is slowed by 
IO, so we want to pre-batch the data to speed things up,
this way reads will be more sequential. 
"""


import numpy as np
import glob
import os
from pathlib import Path
import random
import logging
from simpleemailbot import EmailBot

dataset = "validate"
data_dir = "/data2/cdcm/rescaled"

root_dir = f"{data_dir}/preprocessed/{dataset}"
logging.basicConfig(filename=f'batch-test.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

file_count = 0
paths = []
for root, dirs, files in os.walk(root_dir):
    for name in files:
        if name[-4:] == ".npy":
            file_count += 1
            if file_count % 1000 == 0:
                print(f"Loaded {file_count} paths")
            paths.append(os.path.join(root, name))

logging.debug(f"Num of paths {len(paths)}")
label_dict = {"MR":0, "PT":1,"CT":2,"XR":3}
batch_counter = 0
while len(paths) > 128:
    indices = random.sample(range(0, len(paths)), 128)
    labels = np.zeros(128, dtype=int)
    images = np.zeros((128, 1 , 224, 224), dtype=np.float64)
    temp_paths = []
    slice_start = 0
    for i, selected_index in enumerate(indices):
        image = np.load(paths[selected_index])
        image = np.expand_dims(image, axis=0)
        label_string = Path(paths[selected_index]).parts[-2]
        label=label_dict[label_string]
        labels[i] = label
        images[i] = image
    
    temp_paths = []
    for i in range(0, len(paths)):
        if i not in indices:
            temp_paths.append(paths[i])
    paths = temp_paths
    np.savez(f"{data_dir}/batched/{dataset}/{batch_counter}", x=images,y=labels)
    batch_counter += 1
    logging.debug(f"Processed {batch_counter} batch")
    print(f"Processed {batch_counter} batch")

if len(paths) > 0:  # one more batch to do
    labels = np.zeros(len(paths), dtype=int)
    images = np.zeros((len(paths), 1 , 224, 224), dtype=np.float64)
    temp_paths = []
    slice_start = 0
    for selected_index in range(len(paths)):
        image = np.load(paths[selected_index])
        image = np.expand_dims(image, axis=0)
        label_string = Path(paths[selected_index]).parts[-2]
        label=label_dict[label_string]
        labels[selected_index] = label
        images[selected_index] = image    
    np.savez(f"{data_dir}/batched/{dataset}/{batch_counter}", x=images,y=labels)
    batch_counter += 1

bot = EmailBot("cdcm@st-andrews.ac.uk", "update")
bot.email_me(message=f"finished batching {dataset}")