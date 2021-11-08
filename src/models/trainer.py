"""
Utility functions for training and testing the individual models
"""
import torch
import time
from torch.utils.data import DataLoader
from preprocess.dataset import ImageDataset
from models.cnn import RCNN
from models.loss import MultiTaskLoss
from tqdm import tqdm
from pdb import set_trace as bp

def train(model, trainset, num_epochs=10, lr=0.1, batch_size=2):
    """
    Trainer for the different models on different datasets
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        running_loss = 0.0
        start_time = time.time()

        for i, (img, rois, bbox, cls) in enumerate(tqdm(train_loader)):
            img, rois, bbox, cls = img.to(device), rois.to(device), bbox.to(device), cls.to(device)

            # Concatenate rois
            for batch_i in range(batch_size):

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
        torch.save(model.state_dict(), f"fast-rcnn-epoch{epoch}.pt")
        with open("metrics.txt", 'w') as fp:
            fp.write(f"Epoch {epoch}, Loss: {running_loss/(i + 1)}")

if __name__ == "__main__":
    width = 1280
    height = 720
    n_ch = 3
    model = RCNN(width, height, n_ch)

    trainset = ImageDataset()

    train(model, trainset)
    
    bp()
