
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

def calculate_iou(proposal, label):
    """
    Calculate the intersection over union between bounding boxes of selective search and labels

    :param proposals: A region proposal bounding box
    :param label: Bounding box of label
    """
    x_p, y_p, w_p, h_p = proposal
    x_l, y_l, w_l, h_l = label

    # Get corners of intersection
    x_intersect1, x_intersect2 = max(x_p, x_l), max(x_p + w_p, x_l + w_l)
    y_intersect1, y_intersect2 = max(y_p, y_l), max(y_p + h_p, y_l + h_l)

    # Get areas of intersection
    intersect_width = abs(x_intersect2 - x_intersect1)
    intersect_height = abs(y_intersect2 - y_intersect1)
    intersect_area = intersect_height * intersect_width

    # Calculate union area
    p_area = w_p * h_p
    l_area = w_l * h_l
    union_area = p_area + l_area - intersect_area

    return intersect_area/union_area

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