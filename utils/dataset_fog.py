import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.util import to_tensor, get_anchors, read_class_names
from config import cfg
from utils.augmentation import (
    random_horizontal_flip_fog,
    random_crop_fog,
    random_translate_fog,
    random_horizontal_flip,
    random_crop,
    random_translate,
    image_preprocess,
    resize
)


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


class dataset_fog(Dataset):
    def __init__(self, dataset_type):
        """
        返回
         image(fog/clean):图片 (bs,3,416,416)
         target: 检测框 (num,batch+class+x+y+w+h)
         clean_image: 图片 (bs,416,416,3)

        :param dataset_type: 训练集或测试集？
        """
        super(Dataset, self).__init__()
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        # 输入图片尺寸
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        # 一个批次的图片个数
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        # 图片增强
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG
        # 多尺度
        self.multiscale = cfg.multiscale_training if dataset_type == 'train' else False
        # train or test
        self.data_train_flag = True if dataset_type == 'train' else False
        # 初始时图片大小
        self.train_input_size = 416
        # 数据集中目标的名称
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        # 数据集中目标的个数
        self.num_classes = len(self.classes)
        # yolo中个检测层的锚框个数
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # 3
        # 图片路径和图片中目标的annotation
        self.annotations = np.array(self.load_annotations(dataset_type))[0:10]
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
        image, bboxes, _ = self.parse_annotation(path)
        image = image.permute(2, 0, 1)

        target = preprocess_true_boxes(image, bboxes)

        return path, image, target

    def collate_fn(self, batch):
        self.batch_count += 1

        paths, images, targets = zip(*batch)

        if self.multiscale and self.batch_count % 10 == 0:
            self.train_input_size = random.choice(self.input_sizes)

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
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        img_name = image_path.split('/')[-1]  # 图片名字.jpg
        # print(img_name)
        image_name = img_name.split('.')[0]  # 图片名字
        # print(image_name)
        image_name_index = img_name.split('.')[1]  # 图片格式
        # print('*****************read image***************************')
        # bounding box (num_gt,5) 获得标注
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        # 2/3的概率获得带雾图片
        if random.randint(0, 2) > 0:
            beta = random.randint(0, 9)
            beta = 0.01 * beta + 0.05
            if self.data_train_flag:  # 如果时训练时
                img_name = cfg.vocfog_traindata_dir + image_name \
                           + '_' + ("%.2f" % beta) + '.' + image_name_index
            else:
                img_name = cfg.vocfog_valdata_dir + image_name \
                           + '_' + ("%.2f" % beta) + '.' + image_name_index
            foggy_image = cv2.imread(img_name)
            # 是否对图片进行增强
            if self.data_aug:
                # 1/2 的概率水平反转
                image, foggy_image, bboxes = random_horizontal_flip_fog(image, foggy_image, bboxes)
                # 1/2的概率对图像随机裁剪
                image, foggy_image, bboxes = random_crop_fog(image, foggy_image, bboxes)
                # 1/2的概率进行图像平移
                image, foggy_image, bboxes = random_translate_fog(image, foggy_image, bboxes)
            foggy_image, _ = image_preprocess(np.copy(foggy_image),
                                              [416, 416],
                                              np.copy(bboxes))
            clean_image, bboxes = image_preprocess(np.copy(image),
                                                   [416, 416],
                                                   np.copy(bboxes))
        # 不使用雾天的图片
        else:
            if self.data_aug:
                image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes))
                image, bboxes = random_crop(np.copy(image), np.copy(bboxes))
                image, bboxes = random_translate(np.copy(image), np.copy(bboxes))
            clean_image, bboxes = image_preprocess(np.copy(image),
                                                   [416, 416],
                                                   np.copy(bboxes))
            foggy_image = clean_image

        return to_tensor(foggy_image), bboxes, clean_image

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.switch_backend('TKAgg')
    fog = dataset_fog('train')
    fog_loader = DataLoader(fog, batch_size=2, collate_fn=fog.collate_fn)
    for paths, images, targets in fog_loader:
        print(paths)
        print(images.shape)
        print(targets.shape)
        print(targets[0])
        image = images[0].permute((1, 2, 0))
        w, h = image.shape[0:2]
        # 矩形框的中心点坐标
        center_x, center_y = targets[0][2] * w, targets[0][3] * h
        # 矩形框的宽度和高度
        width, height = targets[0][4] * w, targets[0][5] * h

        # 计算矩形框左下角的坐标
        x = center_x - width / 2
        y = center_y - height / 2
        fig, ax = plt.subplots()
        plt.imshow(image)

        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()
