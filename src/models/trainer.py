"""
Utility functions for training and testing the individual models
"""
import torchvision
from preprocess.preprocess import propose_regions
from pdb import set_trace as bp

def train(model, trainset, valset):
    """
    Trainer for the different models on different datasets
    """
    pass


if __name__ == "__main__":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    bp()