from __future__ import division
import time
import cv2
from utils.util import *
import argparse
import os
import os.path as osp
from model.FY import YoloDIP
import pickle as pkl
import pandas as pd
import random
from utils.util import read_class_names
from config import cfg

def arg_parse():
    """
    设置参数
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    # 检测图像的路径参数
    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default=r".\data\result\before", type=str)
    # 检测后的图像存放路径
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default=r".\data\result\after", type=str)
    # batch_size参数
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    # confidence阈值
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    # IOU阈值
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    # 模型架构文件所在位置
    parser.add_argument("--cfg", dest='cfg_file', help=
    "Config file",
                        default=r".\cfg\yolov3.cfg", type=str)
    # 模型参数所在位置
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default=r"./checkpoint/", type=str)
    # 输入图像的分辨率吧
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


def letterbox_image(img, inp_dim):
    """

    :param img: 输入图片（w,h,c）
    :param inp_dim:输入网络中的图片大小
    :return:
    """
    # 修改图片的尺寸
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)  # （h,w,3）元素全为128

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    预处理图片，输入网络
    """
    img = cv2.resize(img, (inp_dim, inp_dim))  # （w,h,c）
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # (c,w,h)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


# 画出预测框
def write_detection_box(x, results, classes, color):
    c1 = tuple([int(x[1]), int(x[2])])  # x1,x2
    c2 = tuple([int(x[3]), int(x[4])])  # y1,y2
    img = results[int(x[0])]  # int(x[0]):表示来自第几张照片
    cls = int(x[-1])  # 类别index
    label = "{0}".format(classes[cls])  # 类别名
    # 画出检测框矩形框
    cv2.rectangle(img, c1, c2, color, 1)
    # 获取文本的宽高
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # 画出填充矩形框
    cv2.rectangle(img, c1, c2, color, -1)
    # 放置标签
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


def detect():
    args = arg_parse()
    # 待检测图片的路径
    images = args.images
    # batch_size大小
    batch_size = int(args.bs)
    # confidence阈值大小
    confidence = float(args.confidence)
    # IOU阈值大小
    nms_thesh = float(args.nms_thresh)
    start = 0
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    num_classes = cfg.YOLO.CLASSNUM  # coco数据集的类别个数
    classes = read_class_names(cfg.YOLO.CLASSES)

    # 初始化模型
    print("Loading network.....")
    model = YoloDIP(cfg_file=args.cfg_file)

    checkpoint = torch.load(args.weightsfile)
    model.load_state_dict(checkpoint)

    print("Network successfully loaded")
    inp_dim = args.reso

    model.to(device)
    # 设置模型评估模式
    model.eval()

    read_dir = time.time()
    # 检测阶段
    try:
        # images 是存放待检测照片的路径
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        # 如果images不是一个路径，而是一张图片
        imlist = [osp.join(osp.realpath('.'), images)]
    except FileNotFoundError:
        # 如果路径不存在
        print("No file or directory with the name {}".format(images))
        exit()
    # 创建检测后的路径
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    # 读取照片
    load_batch = time.time()
    loaded_ims = [cv2.imread(x) for x in imlist]

    # PyTorch Variables for images
    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

    # 包含原始图片维度的list
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    im_dim_list = im_dim_list.to(device)

    # 创建一个batch
    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    write = 0
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        start = time.time()
        batch = batch.to(device)
        prediction = model(batch)  # 得到预测矩阵（batch_size,(13*13+26*26+52*52)*3,()5+classes）
        prediction = write_results(prediction, confidence, num_classes, device, nms_conf=nms_thesh)
        # (-1,1+4+1+2):第几个batch，四个偏移量，confidence,概率，类别
        end = time.time()
        if type(prediction) == int:
            # 如果batch_size张图片没有检测出目标
            for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue
        # 第i次循环，改变prediction中的第一列（预测框来自第几张图片）
        prediction[:, 0] += i * batch_size

        if not write:
            # i = 0 时
            output = prediction
            write = 1
        else:
            # i =！ 0 时
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 捕捉输入的所有图片是否检测出目标
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long()) / inp_dim
    output[:, 1:5] *= im_dim_list
    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open("./pallete", "rb"))

    draw = time.time()
    # 对output中每一个图片标记上检测框
    list(map(lambda x: write_detection_box(x, loaded_ims, classes, color=random.choice(colors)), output))

    # 保存修改后的图片
    det_names = pd.Series(imlist).apply(lambda x: "{}\\det_{}".format(args.det, x.split("\\")[-1]))
    list(map(cv2.imwrite, det_names, loaded_ims))
    end = time.time()

    # 打印所有步骤的运行时间
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    detect()
