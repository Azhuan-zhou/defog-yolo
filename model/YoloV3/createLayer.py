from __future__ import division
import torch
import torch.nn as nn


def create_modules(blocks):
    """
    创建模型
    :param blocks: 一个元素时字典类型列表，每部字典包含了网络的一个模块
    :return: 模型的信息和模型
    """
    net_info = blocks[0]  # 获取有关输入和预处理的信息
    input_dim = int(net_info['width'])
    module_list = nn.ModuleList()  # 创建一个模型列表
    prev_filters = 3  # 记录卷积的input_channel
    output_filters = []  # 保存每次卷积的output_channel
    filters = None  # 当前卷积核的输出channels
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # 如果模块是卷积
        if x["type"] == "convolutional":
            # 获取激活函数的类型（Linear or LeakyRelu）
            activation = x["activation"]
            try:
                # 获取batch标准化的参数
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            # 获取out_channel的大小
            filters = int(x["filters"])
            # 获取padding_size
            padding = int(x["pad"])
            # 获取卷积核的大小
            kernel_size = int(x["size"])
            # 获取卷积核的步长
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 添加卷积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # 添加batch标准化层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activation_fun = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activation_fun)

        # 如果模块是上采样，使用线性差值的方法
        elif x["type"] == "upsample":
            # 获取上采样的步长
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample)
        # 如果是concatenation或者是计算下一个feature map 需要返回时
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(',')
            # 第一个需要连接的矩阵的索引(-1,end)或者convolutional set 模块的输出矩阵索引(-4)
            start = int(x["layers"][0])
            # 第二个连接的矩阵的索引
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # 使用列表从后索引的方式，去获取需要连接的的输出矩阵
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            # 如果concatenation
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            # 如果时返回之前的输出
            else:
                filters = output_filters[index + start]

        # 残差连接模块
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        # Yolo检测模块
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")  # 某个特征图上像素点3个anchor的（宽，高）
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")  # 获取锚框的宽高
            anchors = [float(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]  # (1,9,2)
            anchors = [anchors[i] for i in mask]  # 获取该特征图的anchor（1,3,2）
            anchors = torch.tensor(anchors, dtype=torch.float32)
            num_class = int(x["classes"])
            stride = int(x['stride'])
            detection = YoloLayer(anchors, num_class, stride)
            module.add_module("Detection_{}".format(index), detection)
        # 一次循环结束，保存模块，更新下一次循环的pre_filters,保存本次循环的filters
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


# yolo检测层
class YoloLayer(nn.Module):
    def __init__(self, anchors, Classes, stride):
        super(YoloLayer, self).__init__()
        self.__anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = Classes
        self.stride = stride

    def forward(self, x):
        # (bs,3*(5+class),grid,grid)
        batch_size, num_grid = x.shape[0], x.shape[-1]
        # (bs, 3, 5+class,grid,grid))
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, num_grid, num_grid).permute(0, 3, 4, 1, 2)

        x_de = self.__decode(x.clone())

        return x, x_de

    def __decode(self, x):
        # (bs,grid,grid,3,5+class)
        batch_size, output_size = x.shape[:2]

        device = x.device
        stride = self.stride
        anchors = self.__anchors.to(device)

        conv_raw_dxdy = x[:, :, :, :, 0:2]  # 中心点位置的偏移 (bs,grid,grid,3,2)
        conv_raw_dwdh = x[:, :, :, :, 2:4]  # 宽高的偏移 (bs,grid,grid,3,2)
        conv_raw_conf = x[:, :, :, :, 4:5]  # anchor的置信度 (bs,grid,grid,3,1)
        conv_raw_prob = x[:, :, :, :, 5:]  # 类别的概率 (bs,grid,grid,3,class)

        y_ = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x_ = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x_, y_], dim=-1)  # 网格坐标(grid,grid,2)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)
        # (bs,grid,grid,3,2)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride  # 预测的坐标: x = [activation(dx) + gridx] * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride  # 预测的宽高： h = [ h * exp(dh) ] * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(conv_raw_conf)  # 预测的置信度

        pred_prob = torch.sigmoid(conv_raw_prob)  # 预测的概率

        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        # (bs,grid,grid,3,5+class) or (bs*grid*grid*3, 5+class)
        return pred_bbox.view(batch_size, -1, 5 + self.num_classes) if not self.training else pred_bbox
