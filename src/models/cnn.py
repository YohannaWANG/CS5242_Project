import cv2
import torch
import torch.nn as nn
from torchvision.ops import RoIPool
from preprocess.preprocess import propose_regions

class RCNN(nn.Module):
    """ Fast Regional Convolutional Neural Network implementation """

    def __init__(self, width, height, n_ch):
        super(RCNN, self).__init__()
        self.convnet = ConvNet(n_ch)

        # Calculate the dimensions after going through self.cnn
        # This will inform scaling factor of ROIPool and in_features size of self.mlp
        test_tensor = torch.rand((1, n_ch, height, width))
        out_tensor = self.convnet(test_tensor)
        _, n_ch, out_height, out_width = out_tensor.size()
        assert height/out_height == width/out_width
        scale_factor = out_height/height
    
        self.roi_pool = RoIPool((7, 7), scale_factor)

        self.classifier = Detector(n_ch * 7 * 7, 2)

    def forward(self, img, roi):
        feature_maps = self.convnet(img)
        roi_out = self.roi_pool(feature_maps, roi)
        roi_out = torch.flatten(roi_out, start_dim=1)
        class_pred, bbox_pred = self.classifier(roi_out)
        return class_pred, bbox_pred

class ConvNet(nn.Module):
    """ The convolution layers for untrained VGG16 """

    def __init__(self, n_ch):
        super(ConvNet, self).__init__()

        # Define the convolution layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, img):
        """
        Forward propagation. img is expected to be a tensor of (bs, n_ch, height, width).
        Intention is a tensor of (bs,) and needs to be one hot encoded to (bs,3).
        """
        feature_maps = self.cnn(img)
        return feature_maps

class Detector(nn.Module):
    """ Neural Network for object detection """

    def __init__(self, input_size, n_class):
        super(Detector, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(in_features=input_size, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )

        # This does classification
        self.classifier = nn.Linear(in_features=4096, out_features=n_class)
        # This does bounding box regression
        self.regressor = nn.Linear(in_features=4096, out_features=n_class * 4)

    def forward(self, x):
        """ Predict class and bounding box for each region """
        x = self.mlp(x)
        class_pred_logits = self.classifier(x)
        bbox_pred_logits = self.regressor(x)
        return class_pred_logits, bbox_pred_logits

if __name__ == "__main__":
    img = cv2.imread("data/Data_GTA/0001.jpg")
    regions = propose_regions(img)

    img_tensor = torch.from_numpy(img).type(torch.float32)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = torch.permute(img_tensor, (0, 3, 1, 2))
    regions_tensor = torch.from_numpy(regions).type(torch.float32)

    _, n_ch, height, width = img_tensor.size()
    
    # Test the RCNN
    model = RCNN(width, height, n_ch)

    # ROI Pooling requires boxes to be of Tensor([K, 5]). First column is img index, which is 0 in this example
    img_idx = torch.zeros((200,1))
    boxes = torch.cat((img_idx, regions_tensor), dim=1)

    class_pred, bbox_pred = model(img_tensor, boxes)