"""
Utility functions for training and testing the individual models
"""
import torch
import time
import argparse
from torch.utils.data import DataLoader
from preprocess.dataset import ImageDataset
from models.cnn import RCNN
from models.mlp import DenseModel
from models.loss import MultiTaskLoss
from models.evaluate import evaluate
from tqdm import tqdm
from pdb import set_trace as bp

def train(model, trainset, testset, model_type='cnn', num_epochs=40, lr=0.1, batch_size=4, num_workers=4):
    """
    Trainer for the different models on different datasets
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        running_loss = 0.0
        start_time = time.time()

        for i, (img, rois, bbox, cls) in enumerate(tqdm(train_loader)):
            img, rois, bbox, cls = img.to(device), rois.to(device), bbox.to(device), cls.to(device)

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
            class_pred, bbox_pred = model(img, cat_rois)
            loss = criterion(class_pred, cat_cls, bbox_pred, cat_bbox)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

            if i % 10 == 0:
                tqdm.write(
                    f"Iteration {i}, Loss: {running_loss/(i + 1)}, Avg iteration time: {(time.time() - start_time)/(i + 1)}s"
                )

        # Save models and metrics
        torch.save(model.state_dict(), f"fast-r{model_type}-epoch{epoch}.pt")
        with open(f"r{model_type}-metrics.txt", 'a+') as fp:
            fp.write(f"Epoch: {epoch}, Loss: {running_loss/(i + 1)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', '-t', default='cnn', type=str)
    args = parser.parse_args()

    width = 1280
    height = 720
    n_ch = 3
    model_type = args.model_type
    num_classes = 6

    trainset = ImageDataset()
    testset = ImageDataset(is_train=False)

    if model_type == 'cnn':
        model = RCNN(width, height, n_ch)
    elif model_type == 'mlp':
        model = DenseModel(width, height, n_ch, num_classes)

    train(model, trainset, testset, model_type=model_type)
    
    bp()
