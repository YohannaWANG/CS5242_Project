"""
Calculate mAP for RCNN performance evaluation
Code by: https://github.com/bes-dev/mean_average_precision
"""
import numpy as np
import pandas as pd
# [xmin, ymin, xmax, ymax, class_id, difficult, crowd]

gt = np.array([
    [439, 157, 556, 241, 0, 0, 0],
    [437, 246, 518, 351, 0, 0, 0],
    [515, 306, 595, 375, 0, 0, 0],
    [407, 386, 531, 476, 0, 0, 0],
    [544, 419, 621, 476, 0, 0, 0],
    [609, 297, 636, 392, 0, 0, 0]
])

# [xmin, ymin, xmax, ymax, class_id, confidence]
preds = np.array([
    [429, 219, 528, 247, 0, 0.460851],
    [433, 260, 506, 336, 0, 0.269833],
    [518, 314, 603, 369, 0, 0.462608],
    [592, 310, 634, 388, 0, 0.298196],
    [403, 384, 517, 461, 0, 0.382881],
    [405, 429, 519, 470, 0, 0.369369],
    [433, 272, 499, 341, 0, 0.272826],
    [413, 390, 515, 459, 0, 0.619459]
])


def _check_empty(preds, gt):
    """ Check empty arguments

    Arguments:
        preds (np.array): predicted boxes.
        gt (np.array): ground truth boxes.

    Returns:
        preds (np.array): predicted boxes.
        gt (np.array): ground truth boxes.
    """
    if not preds.size:
        preds = np.zeros((0, 6))
    if not gt.size:
        gt = np.zeros((0, 7))
    return preds, gt

def _empty_array_2d(size):
    return [[] for i in range(size)]

def compute_iou(pred, gt):
    """ Calculates IoU (Jaccard index) of two sets of bboxes:
            IOU = pred ∩ gt / (area(pred) + area(gt) - pred ∩ gt)

        Parameters:
            Coordinates of bboxes are supposed to be in the following form: [x1, y1, x2, y2]
            pred (np.array): predicted bboxes
            gt (np.array): ground truth bboxes

        Return value:
            iou (np.array): intersection over union
    """
    def get_box_area(box):
        return (box[:, 2] - box[:, 0] + 1.) * (box[:, 3] - box[:, 1] + 1.)

    _gt = np.tile(gt, (pred.shape[0], 1))
    _pred = np.repeat(pred, gt.shape[0], axis=0)

    ixmin = np.maximum(_gt[:, 0], _pred[:, 0])
    iymin = np.maximum(_gt[:, 1], _pred[:, 1])
    ixmax = np.minimum(_gt[:, 2], _pred[:, 2])
    iymax = np.minimum(_gt[:, 3], _pred[:, 3])

    width = np.maximum(ixmax - ixmin + 1., 0)
    height = np.maximum(iymax - iymin + 1., 0)

    intersection_area = width * height
    union_area = get_box_area(_gt) + get_box_area(_pred) - intersection_area
    iou = (intersection_area / union_area).reshape(pred.shape[0], gt.shape[0])
    return iou

def compute_precision_recall(tp, fp, n_positives):
    """ Compute Preision/Recall.

    Arguments:
        tp (np.array): true positives array.
        fp (np.array): false positives.
        n_positives (int): num positives.

    Returns:
        precision (np.array)
        recall (np.array)
    """
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(float(n_positives), 1)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return precision, recall

def compute_match_table(preds, gt, img_id=1 ):
    """ Compute match table.

    Arguments:
        preds (np.array): predicted boxes.
        gt (np.array): ground truth boxes.
        img_id (int): image id

    Returns:
        match_table (pd.DataFrame)


    Input format:
        preds: [xmin, ymin, xmax, ymax, class_id, confidence]
        gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]

    Output format:
        match_table: [img_id, confidence, iou, difficult, crowd]
    """
    def _tile(arr, nreps, axis=0):
        return np.repeat(arr, nreps, axis=axis).reshape(nreps, -1).tolist()

    def _empty_array_2d(size):
        return [[] for i in range(size)]

    match_table = {}
    match_table["img_id"] = [img_id for i in range(preds.shape[0])]
    match_table["confidence"] = preds[:, 5].tolist()
    if gt.shape[0] > 0:
        match_table["iou"] = compute_iou(preds, gt).tolist()
        match_table["difficult"] = _tile(gt[:, 5], preds.shape[0], axis=0)
        match_table["crowd"] = _tile(gt[:, 6], preds.shape[0], axis=0)
    else:
        match_table["iou"] = _empty_array_2d(preds.shape[0])
        match_table["difficult"] = _empty_array_2d(preds.shape[0])
        match_table["crowd"] = _empty_array_2d(preds.shape[0])
    return pd.DataFrame(match_table, columns=list(match_table.keys()))

def row_to_vars(row):
    """ Convert row of pd.DataFrame to variables.

    Arguments:
        row (pd.DataFrame): row

    Returns:
        img_id (int): image index.
        conf (flaot): confidence of predicted box.
        iou (np.array): iou between predicted box and gt boxes.
        difficult (np.array): difficult of gt boxes.
        crowd (np.array): crowd of gt boxes.
        order (np.array): sorted order of iou's.
    """
    img_id = row["img_id"]
    conf = row["confidence"]
    iou = np.array(row["iou"])
    difficult = np.array(row["difficult"])
    crowd = np.array(row["crowd"])
    order = np.argsort(iou)[::-1]
    return img_id, conf, iou, difficult, crowd, order

def check_box(iou, difficult, crowd, order, matched_ind, iou_threshold, mpolicy="greedy"):
    """ Check box for tp/fp/ignore.

    Arguments:
        iou (np.array): iou between predicted box and gt boxes.
        difficult (np.array): difficult of gt boxes.
        order (np.array): sorted order of iou's.
        matched_ind (list): matched gt indexes.
        iou_threshold (flaot): iou threshold.
        mpolicy (str): box matching policy.
                       greedy - greedy matching like VOC PASCAL.
                       soft - soft matching like COCO.
    """
    assert mpolicy in ["greedy", "soft"]
    if len(order):
        result = ('fp', -1)
        n_check = 1 if mpolicy == "greedy" else len(order)
        for i in range(n_check):
            idx = order[i]
            if iou[idx] > iou_threshold:
                if not difficult[idx]:
                    if idx not in matched_ind:
                        result = ('tp', idx)
                        break
                    elif crowd[idx]:
                        result = ('ignore', -1)
                        break
                    else:
                        continue
                else:
                    result = ('ignore', -1)
                    break
            else:
                result = ('fp', -1)
                break
    else:
        result = ('fp', -1)
    return result

def compute_average_precision(precision, recall):
    """ Compute Avearage Precision by all points.

    Arguments:
        precision (np.array): precision values.
        recall (np.array): recall values.

    Returns:
        average_precision (np.array)
    """
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    ids = np.where(recall[1:] != recall[:-1])[0]
    average_precision = np.sum((recall[ids + 1] - recall[ids]) * precision[ids + 1])
    return average_precision

def compute_average_precision_with_recall_thresholds(precision, recall, recall_thresholds):
    """ Compute Avearage Precision by specific points.

    Arguments:
        precision (np.array): precision values.
        recall (np.array): recall values.
        recall_thresholds (np.array): specific recall thresholds.

    Returns:
        average_precision (np.array)
    """
    average_precision = 0.
    for t in recall_thresholds:
        p = np.max(precision[recall >= t]) if np.sum(recall >= t) != 0 else 0
        average_precision = average_precision + p / recall_thresholds.size
    return average_precision

def _evaluate_class(class_id, iou_threshold, recall_thresholds, mpolicy="greedy"):
    """ Evaluate class.

    Arguments:
        class_id (int): index of evaluated class.
        iou_threshold (float): iou threshold.
        recall_thresholds (np.array or None): specific recall thresholds to the
                                              computation of average precision.
        mpolicy (str): box matching policy.
                       greedy - greedy matching like VOC PASCAL.
                       soft - soft matching like COCO.

    Returns:
        average_precision (np.array)
        precision (np.array)
        recall (np.array)
    """
    match_table = compute_match_table(preds, gt)
    table = match_table.sort_values(by=['confidence'], ascending=False)
    matched_ind = {}
    nd = 5 #len(table)
    tp = np.zeros(nd, dtype=np.float64)
    fp = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        img_id, conf, iou, difficult, crowd, order = row_to_vars(table.iloc[d])
        if img_id not in matched_ind:
            matched_ind[img_id] = []
        res, idx = check_box(
            iou,
            difficult,
            crowd,
            order,
            matched_ind[img_id],
            iou_threshold,
            mpolicy
        )
        if res == 'tp':
            tp[d] = 1
            matched_ind[img_id].append(idx)
        elif res == 'fp':
            fp[d] = 1
    precision, recall = compute_precision_recall(tp, fp, class_counter[:, class_id].sum())
    if recall_thresholds is None:
        average_precision = compute_average_precision(precision, recall)
    else:
        average_precision = compute_average_precision_with_recall_thresholds(
            precision, recall, recall_thresholds
        )
    return average_precision, precision, recall

num_classes= 1
iou_thresholds=[0.5]
recall_thresholds = None
mpolicy = "greedy"
preds, gt = _check_empty(preds, gt)

assert preds.ndim == 2 and preds.shape[1] == 6
assert gt.ndim == 2 and gt.shape[1] == 7
class_counter = np.zeros((1, num_classes), dtype=np.int32)

for c in range(num_classes):
    gt_c = gt[gt[:, 4] == c]
    class_counter[0, c] = gt_c.shape[0]
    preds_c = preds[preds[:, 4] == c]
    if preds_c.shape[0] > 0:
        iou = compute_iou(preds, gt).tolist()
    else:
        iou = _empty_array_2d(preds.shape[0])


aps = np.zeros((0, num_classes), dtype=np.float32)
for t in iou_thresholds:
    aps_t = np.zeros((1, num_classes), dtype=np.float32)
    for class_id in range(num_classes):
        aps_t[0, class_id], precision, recall = _evaluate_class(
            class_id, t, recall_thresholds, mpolicy
        )
    aps = np.concatenate((aps, aps_t), axis=0)

print(aps.mean(axis=1).mean(axis=0))

