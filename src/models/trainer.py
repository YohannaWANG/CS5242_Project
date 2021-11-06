"""
Utility functions for training and testing the individual models
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from preprocess.dataset import ImageDataset
from models.cnn import RCNN
from models.loss import MultiTaskLoss
from tqdm import tqdm
from pdb import set_trace as bp

def train(model, trainset, num_epochs=1, lr=0.1, batch_size=2):
    """
    Trainer for the different models on different datasets
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        for i, (img, rois, labels) in enumerate(tqdm(train_loader)):
            img, rois, labels = img.to(device), rois.to(device), labels.to(device)

            # Concatenate rois
            for batch_i in range(batch_size):

                indices = torch.ones((rois.size(1),1)) * batch_i
                # rois = torch.cat((indices, rois), dim=1)
                if batch_i == 0:
                    cat_rois = rois[batch_i]
                    cat_indices = indices
                    cat_labels = labels[batch_i]
                else:
                    cat_rois = torch.cat((cat_rois, rois[batch_i]))
                    cat_indices = torch.cat((cat_indices, indices))
                    cat_labels = torch.cat((cat_labels, labels[batch_i]))

            cat_rois = torch.cat((cat_indices, cat_rois), dim=1)
            class_pred, bbox_pred = model(img, cat_rois)
            loss = criterion(class_pred, cat_labels[:,4], bbox_pred, cat_labels[:,:4])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm.write(f"Iteration {i}, Loss: {loss}")

if __name__ == "__main__":
    width = 1280
    height = 720
    n_ch = 3
    model = RCNN(width, height, n_ch)

    trainset = ImageDataset()

    train(model, trainset)
    
    bp()