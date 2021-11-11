import torch
import torch.nn as nn
from torchvision.ops import RoIPool

class DenseModel(nn.Module):
    """ The custom model for Spot robot """

    def __init__(self, width, height, n_ch, n_class):
        super(DenseModel, self).__init__()

        # ROI max pooling on image itself
        self.roi_pool = RoIPool((224, 224), 1)

        # Get a pseudo size
        input_size = n_ch * 224 * 224

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(in_features=input_size, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        
        # This does classification
        self.classifier = nn.Linear(in_features=1024, out_features=n_class)
        # This does bounding box regression
        self.regressor = nn.Linear(in_features=1024, out_features=4)

    def forward(self, img, roi):
        """
        Forward propagation. img is expected to be a tensor of (bs, n_ch, height, width).
        Intention is a tensor of (bs,) and needs to be one hot encoded to (bs,3).
        """
        roi_out = self.roi_pool(img, roi)
        x = torch.flatten(roi_out, start_dim=1)
        logits = self.mlp(x)
        class_pred_logits = self.classifier(logits)
        bbox_pred_logits = self.regressor(logits)
        return class_pred_logits, bbox_pred_logits