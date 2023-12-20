from __future__ import division

import time

import torch
import numpy as np
import torchvision
from tqdm import tqdm


def unique(tensor):
    # 获取一个tensor不重复的值
    tensor_cpu = tensor.cpu()
    tensor_np = tensor_cpu.numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


# NMS(非极大值抑制)
def write_results(prediction, confidence, num_classes, device, nms_conf=0.4):
    """
    进行非极大值抑制
    :param prediction: 模型预测的数据（batch_size,(13*13+26*26+52*53)*3,(5+classes)）
    :param confidence:置信度阈值
    :param num_classes: 类别的个数
    :param nms_conf:设定IOU的阈值
    :return:预测的检测框
    """
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # 返回一个布尔数组，元素大于confidence,取True;否则取False;un_squeeze在dim=2增加一个维度
    prediction = prediction * conf_mask  # 获取置信度满足要求的anchor (batch_size,anchors,5+classes),否则anchor的所有数据置零
    # anchors = 13*13+26*26+52*53)*3
    # 从(x,y,w,h)转化为(x1,y1,x2,y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)  # 左上角x:x-w/2
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)  # 左上角y:y-h/2
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)  # 右下角x:x+w/2
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)  # 左下角y:y+h/2
    prediction[:, :, :4] = box_corner[:, :, :4]
    batch_size = prediction.size(0)
    write = False
    for ind in range(batch_size):
        image_pred = prediction[ind]  # 一个batch表示的是一张图片的所有anchor (anchors,5+classes)
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # max_conf是最大概率，max_conf_score是最大概率所在位置（anchor所属的类别）；size都是（anchors）
        max_conf = max_conf.float().unsqueeze(1)  # (anchors,1)
        max_conf_score = max_conf_score.float().unsqueeze(1)  # (anchors,1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)  # (anchors,5+1+1)
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))  # confidence不等于0的行
        # 摆脱confidence不满足要求的anchor
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)  # (non_zero_ind_size,5+1+1)
        except:
            continue
        if image_pred_.shape[0] == 0:
            # 如果一张图像中没有检测到任何目标
            continue
        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1]).to(device)  # 传入一张图片所有anchor的概率（non_zero_ind_size），得到这张图片有那些目标类别
        for cls in img_classes:
            # 对类别逐个进行NMS
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()  # 最大预测概率为cls类别的anchor的mask
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)  # 筛选出cls类别的anchor （idx,7）
            # 对image_pred_class的confidence（4的位置）排序
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]  # 排序后的index
            image_pred_class = image_pred_class[conf_sort_index]  # 使用index索引对image_pred_class 排序 (idx,7)
            idx = image_pred_class.size(0)  # image_pred_class 中anchor的个数
            for i in range(idx):
                # 检测框i和检测框i之后的检测框计算IOU
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:], x1y1x2y2=True)
                except ValueError:
                    # 当image_pred_class中第i个位置已经是最后一个元素时，i+1会切出一个空的tensor
                    break
                except IndexError:
                    # 当i超出image_pred_class的索引范围时，会报错
                    break
                # IOU大于设定阈值，则被抑制
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask  # 计算保留下来的检测框，被抑制的检测框所在的列被设置为0
                # 将被抑制的检测框删去，剩下D个检测框
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)  # （D,7）
            # batch_ind是一个包含mage_pred_class.size(0)个元素的张量，其元素值为检测框是第几张图片的
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)  # (D,1)
            seq = batch_ind, image_pred_class

            if not write:
                # 当为第一张图片的时候
                output = torch.cat(seq, 1)  # （D,1+7）
                write = True
            else:
                # 不为第一张图片的时候
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    :param box1:维度为(num_objects, 4)
    :param box2:维度为(num_objects, 4)
    :param x1y1x2y2: 表示输入的目标框是否为上下角点坐标
    :return:
    """
    if isinstance(box1, np.ndarray):
        box1 = torch.tensor(box1, dtype=torch.float32)
        box2 = torch.tensor(box2, dtype=torch.float32)
    # 获得边框左上角点和右下角点的坐标
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 计算真实框与预测框的交集矩形的左上角点和右下角点的坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area交集面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  # box1的面积
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)  # box2的面积

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)  # 计算交并比

    return iou


def bbox_wh_iou(wh1, wh2):
    """
    计算 Ground truth 和 anchor 的IOU
    :param wh1: box1的宽高，维度为(2, )，即一维数组
    :param wh2: box2的宽高，维度为(num_objects, 2)，第一列是宽，第二列是高
    :return: 返回交并比，维度为(num_objects, )，即一维数组
    """
    wh2 = wh2.t()
    # 转置之后，wh2的维度为(2, num_objects)，
    # 这样就能保证wh2[0]是宽, wh2[1]是高

    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)  # 计算交集
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area  # 计算并集

    # 返回交并比
    return inter_area / union_area




def to_cpu(tensor):
    return tensor.detach().cpu()


def to_tensor(ndarray):
    return torch.tensor(ndarray, dtype=torch.float32)


def read_class_names(class_file_name):
    '''
    从文件中加载目标名称
    :param class_file_name:
    :return:
    '''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape((3, 3, 2))


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics



def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap