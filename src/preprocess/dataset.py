"""
Pytorch dataset for use on dataloaders
"""
import os
import cv2
import pandas as pd
from preprocess.preprocess import propose_regions
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    Custom Dataset class to load images and propose regions using selective search
    to prepare for CNN/MLP training.
    """

    def __init__(self, data_dir="data/Data_GTA", annotation_file="data/boxes-clean.csv", seed=1, is_train=True):
        self.data_dir = data_dir
        self.image_files = [name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))]
        self.is_train = is_train

        self.annotations = pd.read_csv(annotation_file)

    def __len__(self):
        return len(self.num_images)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.image_files[index])
        img = cv2.imread("img_path")
        regions =  propose_regions(img, num_regions=1000)
        annotations = self.annotations[self.annotations["filename"] == self.image_files[index]]

        return img_path, regions, annotations