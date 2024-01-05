import numpy as np
import os
import cv2
import math
from tqdm import tqdm


# only use the image including the labeled instance objects for training
def load_annotations(annot_path):
    print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations


# print('*****************Add haze offline***************************')
def parse_annotation(annotation):
    line = annotation.split()
    image_path = line[0]
    # print(image_path)
    img_name = image_path.split('/')[-1]
    # print(img_name)
    image_name = img_name.split('.')[0]
    # print(image_name)
    image_name_index = img_name.split('.')[1]
    # print(image_name_index)

    # '../data/data_fog/train/JPEGImages/'
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)
    image = cv2.imread(image_path)
    times = 0
    for i in range(10):
        # 增加雾
        def AddHaz_loop(img_f, center, size, beta, A):
            (row, col, chs) = img_f.shape

            for w in range(row):
                for h in range(col):
                    d = -0.04 * math.sqrt((w - center[0]) ** 2 + (h - center[1]) ** 2) + size
                    td = math.exp(-beta * d)
                    img_f[w][h][:] = img_f[w][h][:] * td + A * (1 - td)  # 根据大气散射模型
            return img_f

        img_f = image / 255
        (row, col, chs) = image.shape
        A = 0.5
        # beta = 0.08
        beta = 0.01 * i + 0.05
        size = math.sqrt(max(row, col))
        center = (row // 2, col // 2)
        # 对图像加雾
        foggy_image = AddHaz_loop(img_f, center, size, beta, A)
        # 控制范围在（0，255）
        img_f = np.clip(foggy_image * 255, 0, 255)
        img_f = img_f.astype(np.uint8)
        #img_name = './data/data_fog/train/JPEGImages/' + image_name \
                   #+ '_' + ("%.2f" % beta) + '.' + image_name_index
        img_name = './data/data_fog/test/JPEGImages/' + image_name \
          + '_' + ("%.2f" % beta) + '.' + image_name_index
        cv2.imwrite(img_name, img_f)


if __name__ == '__main__':
    # 训练数据
    #an = load_annotations('./data/data_fog/voc_norm_train.txt')
    # 测试数据
    an = load_annotations('data/data_fog/voc_norm_test.txt')
    ll = len(an)
    print(ll)
    for j in tqdm(range(len(an))):
        print('processing {}:'.format(an[j].strip().split()[0]))
        parse_annotation(an[j])
