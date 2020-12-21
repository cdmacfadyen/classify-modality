import torch
from torchvision import datasets
import numpy as np
import random
import os
import torch.utils.data as data
import time
import logging

logging.getLogger(__name__)
class ImageAndPathDataset(datasets.DatasetFolder):
    """Custom pytorch dataset that extends DataSetFolder
    to also return the path of the image for extra analysis. 
    I want to do this because I have a metadata csv that maps 
    the path to dataset it came from and its original 
    image file.
    """

    def __getitem__(self, index):
        image_label_tuple = super(ImageAndPathDataset, self).__getitem__(index)
        path_to_image = self.samples[index][0]
        tuple_with_path = image_label_tuple + (path_to_image,)   # extends tuple
        return tuple_with_path

def npy_loader(path):
    img = torch.from_numpy(np.load(path))
    img = img.unsqueeze(0)
    return img


random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.benchmark = True

class NpyDataLoader():
    def __init__(self, data_root, batch_size=32):
        self._train_dataset = ImageAndPathDataset(
            root = f"{data_root}/preprocessed/train",
            loader=npy_loader,
            extensions=(".npy",)
        )

        self.train_loader = torch.utils.data.DataLoader(self._train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=16,
            pin_memory=True,
            )

        self._validate_dataset = ImageAndPathDataset(
            root = f"{data_root}/preprocessed/validate",
            loader=npy_loader,
            extensions=(".npy",)
        )

        self.validate_loader = torch.utils.data.DataLoader(self._validate_dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=16,
            pin_memory=True,
        )

        self.classes = self._validate_dataset.classes
        self.batch_size = batch_size

class BatchedDataset(data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.length = len(os.listdir(data_root))
    def __getitem__(self, index):
        # load_start = time.time()
        npz_dict = np.load(f"{self.data_root}/{index}.npz")
        image = npz_dict["x"]
        label = npz_dict["y"]
        # load_end = time.time()
        # print(f"\tLoad time: {load_end - load_start}")
        return torch.from_numpy(image).float(), torch.from_numpy(label), ""

    def __len__(self):
        return self.length
