import torch
import torch.nn as nn

class Model(nn.Module):
    """ The custom model for Spot robot """

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(in_features=input_size, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=output_size),
        )

    def forward(self, img):
        """
        Forward propagation. img is expected to be a tensor of (bs, n_ch, height, width).
        Intention is a tensor of (bs,) and needs to be one hot encoded to (bs,3).
        """
        X = torch.flatten(img, start_dim=1)
        logits = self.mlp(X)
        return logits