"""
Custom Loss Functions for Object Detection
"""
import torch
import torch.nn as nn
from pdb import set_trace as bp

class MultiTaskLoss(nn.Module):

    def __init__(self, alpha=1.0):
        """
        Constructor for multi-task loss

        :param alpha: This is actually the lambda hyperparameter that controls 
                    the weight between classification and regression loss.
        """
        super(MultiTaskLoss, self).__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.SmoothL1Loss()
        self.alpha = alpha

    def forward(self, logits, labels, bbox_pred, bbox_labels):
        """
        Compute the multi-task loss function.

        :param logits: class predictions
        :param labels: ground truth labels of class predictions
        :param bbox_pred: bounding box predictions
        :param bbox_labels: bounding box labels
        """
        L_cls = self.cls_criterion(logits, labels)
        L_loc = self.regression_criterion(bbox_pred, bbox_labels)
        iverson_indicator = (labels > 0).type(torch.LongTensor)

        # Calclutate multi-task loss
        loss = L_cls + (self.alpha * iverson_indicator * L_loc).sum()

        return loss