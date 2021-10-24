import torch
import torch.nn as nn

class Model(nn.Module):
    """ The custom model for Spot robot """

    def __init__(self, width, height, n_ch, output_size):
        super(Model, self).__init__()

        # Define the convolution layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calculate the dimensions after going through self.cnn
        # This will inform in_features size of self.mlp
        test_tensor = torch.rand((1, n_ch, height, width))
        out_tensor = self.cnn(test_tensor)
        _, out_ch, out_height, out_width = out_tensor.size()

        # Dimension is flattened output image and an ohe of the intention
        in_features = out_width * out_height * out_ch

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features=in_features, out_features=4096),
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
        X = self.cnn(img)
        X = torch.flatten(X, start_dim=1)
        logits = self.mlp(X)
        return logits