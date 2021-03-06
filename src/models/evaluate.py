"""
Script for evaluating trained models
"""

import torch
from torch.utils.data import DataLoader
from mean_average_precision import MetricBuilder

import argparse
import numpy as np
from tqdm import tqdm

from models.cnn import RCNN
from models.mlp import DenseModel
from preprocess.mAP import average_precision
from preprocess.dataset import ImageDataset

from pdb import set_trace as bp

def evaluate(model, testset, batch_size=4, num_workers=4):
    """
    Evaluates the mean average precision (mAP) of the model given ground truth.

    :param model: A pytorch model to evaluate.
    :param testset: Test set data for evaluation. A pytorch dataset object.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)

    running_mean = 0.0

    for i, (img, rois, bbox, cls) in enumerate(tqdm(test_loader)):
        img, rois, bbox, cls = img.to(device), rois.to(device), bbox.to(device), cls.to(device)
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=len(cls.unique()))

        # Concatenate rois
        for batch_i in range(img.size(0)):

            indices = torch.ones((rois.size(1),1)) * batch_i
            indices = indices.to(device)
            # rois = torch.cat((indices, rois), dim=1)
            if batch_i == 0:
                cat_rois = rois[batch_i]
                cat_indices = indices
                cat_bbox = bbox[batch_i]
                cat_cls = cls[batch_i]
            else:
                cat_rois = torch.cat((cat_rois, rois[batch_i]))
                cat_indices = torch.cat((cat_indices, indices))
                cat_bbox = torch.cat((cat_bbox, bbox[batch_i]))
                cat_cls = torch.cat((cat_cls, cls[batch_i]))

        cat_rois = torch.cat((cat_indices, cat_rois), dim=1)

        with torch.no_grad():
            class_pred, bbox_pred = model(img, cat_rois)
            class_pred_cls = torch.argmax(class_pred, 1).unsqueeze(1).type(torch.LongTensor)
            class_pred_score = torch.max(torch.softmax(class_pred, dim=1), 1)[0].unsqueeze(1)
            class_pred_cls = class_pred_cls.to(device)
            class_pred_score = class_pred_score.to(device)

            # Prepare preds and ground truth for average precision
            preds = torch.cat((torch.cat((bbox_pred, class_pred_cls), dim=1), class_pred_score), dim=1)
            gt = torch.cat((cat_bbox, cat_cls.unsqueeze(1)), dim=1)

            # Mask of not background in ground truth
            # TODO
            gt_mask = (cat_cls.unsqueeze(1) > 0)

            # Add some extra dimensions for average precision
            difficult = torch.zeros((gt.size(0),1), dtype=torch.float32).to(device)
            gt = torch.cat((gt, difficult), 1)
            crowd = torch.zeros((gt.size(0),1), dtype=torch.float32).to(device)
            gt = torch.cat((gt, crowd), 1)

            # Use only the non background to calculate mAP
            # preds = torch.masked_select(preds, gt_mask)
            # gt = torch.masked_select(gt, gt_mask)

        preds, gt = preds.cpu().numpy(), gt.cpu().numpy()

        metric_fn.add(preds, gt)
        running_mean += metric_fn.value(iou_thresholds=0.5)['mAP'].mean()

    print(f"Total mAP: {running_mean/(i+1)}")

    return running_mean/(i+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', '-t', default='cnn', type=str)
    args = parser.parse_args()

    width = 1280
    height = 720
    n_ch = 3
    model_type = args.model_type
    num_classes = 6
    num_epoch = 38

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type == "cnn":
        model = RCNN(width, height, n_ch)
    elif model_type == "mlp":
        model = DenseModel(width, height, n_ch, num_classes)
    else:
        raise NotImplementedError(f"{model_type} is not an implemented model type!")

    # Load the model trained for each epoch
    for epoch in range(num_epoch):
        model.load_state_dict(torch.load(f"trained_models/fast-r{model_type}-epoch{epoch}.pt", map_location=device))
        testset = ImageDataset(is_train=False)
        score = evaluate(model, testset)
        with open(f"r{model_type}-mAP.txt", 'a+') as fp:
            fp.write(f"Epoch {epoch} score: {score}\n")
