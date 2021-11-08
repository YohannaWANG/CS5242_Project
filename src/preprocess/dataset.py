"""
Pytorch dataset for use on dataloaders
"""
import os
import cv2
import torch
import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from preprocess.preprocess import propose_regions, calculate_iou
from pdb import set_trace as bp

class ImageDataset(Dataset):
    """
    Custom Dataset class to load images and propose regions using selective search
    to prepare for CNN/MLP training.
    """

    def __init__(self, data_dir="data/Data_GTA", annotation_file="data/boxes-clean.csv", is_train=True):
        self.data_dir = data_dir
        self.image_files = [name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))]
        self.is_train = is_train

        self.annotations = pd.read_csv(annotation_file)

        self.classes = {
            "background": 0,
            "car": 1,
            "trafficlight": 2,
            "bus": 3,
            "truck": 4,
            "person": 5
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.image_files[index])
        img = cv2.imread(img_path)
        annotations = self.annotations[self.annotations["Filename"] == self.image_files[index]]
        
        # Get regions of interests
        regions = propose_regions(img, num_regions=2000)
        zeros = np.zeros((len(regions), 2), dtype=float)
        labels = np.concatenate((copy.deepcopy(regions), zeros), axis=1)

        # Calculate iou of each roi with annotations
        # Assign the annotation that has the largest iou as the bbox label
        for idx, region in enumerate(regions):
            for _, row in annotations.iterrows():
                bbox_width = row['Right'] - row['Left']
                bbox_height = row['Bottom'] - row['Top']
                # iou = calculate_iou(tuple(region), (row['Left'], row['Top'], bbox_width, bbox_height))
                iou1 = calculate_iou(
                    (region[0], region[1], region[0] + region[2], region[1] + region[3]), 
                    (row['Left'], row['Top'], row['Right'], row['Bottom'])
                )

                iou2 = calculate_iou(
                    (row['Left'], row['Top'], row['Right'], row['Bottom']),
                    (region[0], region[1], region[0] + region[2], region[1] + region[3]) 
                )

                iou = max(iou1, iou2)

                if iou >= 0.5 and iou > labels[idx][5]:
                    # Found an object bounding box roi
                    labels[idx][0] = row['Left']
                    labels[idx][1] = row['Top']
                    labels[idx][2] = bbox_width
                    labels[idx][3] = bbox_height
                    labels[idx][4] = self.classes.get(row['Label'], 0)
                    labels[idx][5] = iou
        

        img_tensor = torch.from_numpy(img).type(torch.FloatTensor).permute(2,0,1)
        roi_tensor = torch.from_numpy(regions).type(torch.FloatTensor)
        bbox_tensor = torch.from_numpy(labels[:,:4]).type(torch.FloatTensor)
        cls_tensor = torch.from_numpy(labels[:,4]).type(torch.LongTensor)

        # ROI Pooling requires boxes to be of Tensor([K, 5]). First column is img index, which is 0 in this example
        # img_idx = torch.ones((roi_tensor.size(0),1)) * index
        # roi_tensor = torch.cat((img_idx, roi_tensor), dim=1)

        return img_tensor, roi_tensor, bbox_tensor, cls_tensor

if __name__ == "__main__":
    dataset = ImageDataset()
    print(f"Dataset contains {len(dataset)} images")
    # Take a sample for sanity
    img, labels = dataset[0]
    bp()