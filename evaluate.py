from __future__ import division

import tqdm
import numpy as np

from terminaltables import AsciiTable
from utils.utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
import torch
from torch.utils.data import DataLoader


def _evaluate(model, dataloader, device, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    for _, images, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        images = images.to(device)

        targets = targets
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        with torch.no_grad():
            predictions, _ = model(images)
            outputs = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


if __name__ == '__main__':
    from model.FY import YoloDIP
    from utils.utils import read_class_names
    from dataset_fog import dataset_fog
    from torch.utils.data import DataLoader
    from config import cfg
    from train import eval_epoch

    device = torch.device('cuda')
    model = YoloDIP(cfg_file=cfg.model_def).to(device)
    weight = './checkpoints/exp03/FY_ckpt_4.pth'
    model.load_state_dict(torch.load(weight, map_location=device))
    print('using .pth init model')

    class_names = read_class_names('./data/classes/vocfog.names')

    dataset_eval = dataset_fog('test')

    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.n_cpu,
        collate_fn=dataset_eval.collate_fn
    )

    _ = eval_epoch(model, dataloader_eval, device, class_names)
