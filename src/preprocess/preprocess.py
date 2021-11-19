
import cv2
import os
import copy
import numpy as np
import torch
import pandas as pd
from pdb import set_trace as bp

def propose_regions(img, num_regions=200):
    """
    Implementation of selective search for region proposal.
    We use SelectivSearchFast because we try not to bottleneck downstream neural network.

    :param img: Input image.
    :param num_regions: Number of top regions to return. Default is 200.
    """
    segmentor = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    segmentor.setBaseImage(img)
    segmentor.switchToSelectiveSearchFast()
    regions = segmentor.process()

    # Filter regions that are too small
    width, height, _ = img.shape
    filter = []
    for _, _, region_width, region_height in regions:
        filter.append((float(region_width)/width >= 0.1 or float(region_height)/height >= 0.1))

    regions = regions[filter]

    return regions[:num_regions]

def visualize_regions(img, regions):
    """
    Visualize region proposals on an img.

    :param img: Input image.
    :param regions: Region proposals output from propose_regions function.
    """
    # Don't draw on original image
    img_copy = img.copy()
    for x, y, width, height in regions:
        cv2.rectangle(img_copy, (x, y), (x + width, y + height), (0, 255, 0), 1, cv2.LINE_AA)

    return img_copy

def calculate_iou(proposals, label):
    """
    Calculate the intersection over union between bounding boxes of selective search and labels
    This code is adapted from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

    :param proposals: A region proposal bounding box
    :param label: Bounding box of label
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(proposals[0], label[0])
    yA = max(proposals[1], label[1])
    xB = min(proposals[2], label[2])
    yB = min(proposals[3], label[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    proposalsArea = abs((proposals[2] - proposals[0]) * (proposals[3] - proposals[1]))
    labelArea = abs((label[2] - label[0]) * (label[3] - label[1]))
    unionArea = proposalsArea + labelArea - interArea

    iou = interArea / float(unionArea)

    return iou

if __name__ == "__main__":
    # Preprocess all the images
    data_dir = "data/Data_GTA/"
    files = os.listdir(data_dir)
    all_annotations = pd.read_csv("data/boxes-clean.csv")

    classes = {
        "background": 0,
        "car": 1,
        "trafficlight": 2,
        "bus": 3,
        "truck": 4,
        "person": 5
    }

    # Initialize tensors
    roi = torch.zeros((1, 5)).type(torch.LongTensor)
    gt = torch.zeros((1, 5)).type(torch.LongTensor)


    for idx, file in enumerate(files):
        # Read image
        print(f"Processing {file}")
        filepath = os.path.join(data_dir, file)
        img = cv2.imread(filepath)

        # Propose regions
        regions = propose_regions(img, num_regions=2000)
        # proposal_img = visualize_regions(img, regions)
        zeros = np.zeros((len(regions), 2), dtype=float)
        labels = np.concatenate((copy.deepcopy(regions), zeros), axis=1)

        # Get annotations of this image
        annotations = all_annotations[all_annotations["Filename"] == file]

        roi_tensor = torch.from_numpy(regions).type(torch.LongTensor)
        label_tensor = torch.from_numpy(labels[:,:-1]).type(torch.LongTensor)

        # ROI Pooling requires boxes to be of Tensor([K, 5]). First column is img index, which is 0 in this example
        img_idx = torch.ones((roi_tensor.size(0),1)) * idx
        roi_tensor = torch.cat((img_idx, roi_tensor), dim=1)

        if idx == 0:
            roi = roi_tensor
        else:
            roi = torch.cat((roi, roi_tensor), dim=0)

    torch.save(roi, "data/roi.pkl")

    bp()