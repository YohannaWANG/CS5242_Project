
import cv2
from pdb import set_trace as bp

def propose_regions(img):
    """
    Implementation of selective search for region proposal.
    We use SelectivSearchFast because we try not to bottleneck downstream neural network.

    :param img: Input image.
    """
    segmentor = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    segmentor.setBaseImage(img)
    segmentor.switchToSelectiveSearchQuality()
    regions = segmentor.process()

    return regions

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


if __name__ == "__main__":
    img = cv2.imread("data/Data_GTA/0001.jpg")
    regions = propose_regions(img)
    proposal_img = visualize_regions(img, regions)
    cv2.imshow("Proposals", proposal_img)
    cv2.waitKey(0)
    bp()