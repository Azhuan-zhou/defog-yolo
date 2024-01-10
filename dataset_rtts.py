import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import cfg

from utils.augmentation import (
    image_preprocess,
    resize
)
from utils.utils import to_tensor, read_class_names


def preprocess_true_boxes(image, bboxes):
    w, h = image.shape[1:]
    target = []
    for bbox in bboxes:  # 对一个图片中的每一个gt

        bbox_coor = bbox[:4]  # 左上角和右下角坐标 (4,)
        bbox_class_ind = int(bbox[4])  # 标签

        # xyxy转化为xywh
        # x0 = ( x1 + x2 ) / 2 ; y = ( y1 + y2 ) / 2  ; w = ( x2 - x1 ) / 2 ; h = ( y2 - y1 ) / 2
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh[[0, 2]] = bbox_xywh[[0, 2]] / w  # 归一化
        bbox_xywh[[1, 3]] = bbox_xywh[[1, 3]] / h

        target_b = torch.zeros(1, 6)
        target_b[0, 2:] = torch.tensor(bbox_xywh, dtype=torch.float32)
        target_b[0, 1] = torch.tensor(bbox_class_ind, dtype=torch.float32)
        target.append(target_b)

    return torch.cat(target, dim=0)

class dataset_rtts(Dataset):
    def __init__(self, dataset_type):
        """
        返回
         image(fog/clean):图片 (bs,3,416,416)
         target: 检测框 (num,batch+class+x+y+w+h)
         clean_image: 图片 (bs,416,416,3)

        :param dataset_type: 训练集或测试集？
        """
        super(Dataset, self).__init__()
        self.annot_path = cfg.RTTS.path
        # 输入图片尺寸
        self.input_sizes = cfg.RTTS.INPUT_SIZE
        # 一个批次的图片个数
        self.batch_size = cfg.RTTS.BATCH_SIZE 
        # train or test
        self.data_train_flag = False
        # 初始时图片大小
        self.train_input_size = 416
        # 数据集中目标的名称
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        # 数据集中目标的个数
        self.num_classes = len(self.classes)
        # yolo中个检测层的锚框个数
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # 3
        # 图片路径和图片中目标的annotation
        self.annotations = np.array(self.load_annotations(dataset_type))
        # 样本个数
        self.num_samples = len(self.annotations)

        self.batch_count = 0

    # only use the image including the labeled instance objects for training
    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]  # 有标注的图片
        # np.random.shuffle(annotations)
        print('###################the total image:', len(annotations))
        return annotations

    def __getitem__(self, index):
        path = self.annotations[index]
        line = path.split()
        image_path = line[0]
        image, bboxes, _ = self.parse_annotation(path)
        image = image.permute(2, 0, 1)

        target = preprocess_true_boxes(image, bboxes)

        return image_path, image, target

    def collate_fn(self, batch):
        self.batch_count += 1

        paths, images, targets = zip(*batch)

        images = torch.stack([resize(img, self.train_input_size) for img in images])

        tensor_targets = []
        for i, target in enumerate(targets):
            target[:, 0] = i
            tensor_targets.append(target)

        return paths, images, torch.cat(tensor_targets, dim=0)

    def parse_annotation(self, annotation):
        """
        对输入的图片和标注主力
        :param annotation: （图片路径，标注）
        :return: image()
        """
        line = annotation.split()
        image_path = '.'+line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        img_name = image_path.split('/')[-1]  # 图片名字.jpg
        image_name = img_name.split('.')[0]  # 图片名字
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        clean_image, bboxes = image_preprocess(np.copy(image),
                                               [416, 416],
                                               np.copy(bboxes))
        foggy_image = clean_image

        return to_tensor(foggy_image), bboxes, clean_image

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
    from utils.plot import plot
    from torchvision.ops import box_convert

    plt.switch_backend('TKAgg')

    # fog = dataset_fog('test')
    # fog_loader = DataLoader(fog, batch_size=1, collate_fn=fog.collate_fn)
    rtts_dataset = dataset_rtts('test')
    rtts_loader = DataLoader(rtts_dataset,batch_size=1,collate_fn=rtts_dataset.collate_fn)
    for paths, images, targets in rtts_loader:
        print(paths)
        image = images[0] * 255
        class_names = targets[:, 1]
        class_names = [rtts_dataset.classes[int(i)] for i in class_names]
        bboxs = targets[:, 2:]
        bboxs = box_convert(bboxs, 'cxcywh', 'xyxy')
        bboxs = bboxs * image.shape[-1]
        bboxs = BoundingBoxes(bboxs, format='xyxy', canvas_size=image[-2:])
        plot([(image, bboxs, class_names)])
        plt.show()
