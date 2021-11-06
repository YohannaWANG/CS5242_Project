
import cv2
import torch
from torchvision.ops import RoIPool
from pdb import set_trace as bp

def propose_regions(img, num_regions=200):
    """
    Implementation of selective search for region proposal.
    We use SelectivSearchFast because we try not to bottleneck downstream neural network.

    :param img: Input image.
    :param num_regions: Number of top regions to return. Default is 200.
    """
    segmentor = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    segmentor.setBaseImage(img)
    segmentor.switchToSelectiveSearchFast()
    regions = segmentor.process()

    # Filter regions that are too small
    width, height, _ = img.shape
    filter = []
    for _, _, region_width, region_height in regions:
        filter.append((float(region_width)/width >= 0.1 or float(region_height)/height >= 0.1))

    regions = regions[filter]

    return regions[:num_regions]

def visualize_regions(img, regions):
    """
    Visualize region proposals on an img.

    :param img: Input image.
    :param regions: Region proposals output from propose_regions function.
    """
    # Don't draw on original image
    img_copy = img.copy()
    for x, y, width, height in regions:
        cv2.rectangle(img_copy, (x, y), (x + width, y + height), (0, 255, 0), 1, cv2.LINE_AA)

    return img_copy

def calculate_iou(boxA, boxB):
    """
    Calculate the intersection over union between bounding boxes of selective search and labels

    :param proposals: A region proposal bounding box
    :param label: Bounding box of label
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

if __name__ == "__main__":
    img = cv2.imread("data/Data_GTA/0001.jpg")
    regions = propose_regions(img)
    proposal_img = visualize_regions(img, regions)

    # cv2.imshow("Proposals", proposal_img)
    # cv2.waitKey(0)

    regions_tensor = torch.from_numpy(regions)
    roipool_layer = RoIPool((7,7), 1/16)
    derp = roipool_layer(regions_tensor)
    bp()