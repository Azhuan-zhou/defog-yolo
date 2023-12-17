from __future__ import division

import torch
import torch.nn as nn

from model.model_config.parse_yolov3_config import parse_cfg
from utils.util import *
from model.YoloV3.createLayer import create_modules


class YoloV3(nn.Module):
    def __init__(self, cfg_file=r'./yolov3.cfg'):
        super(YoloV3, self).__init__()
        self.device = None
        self.seen = None  # 模型的参数量
        self.header = None  # 模型的徒步信息
        self.blocks = parse_cfg(cfg_file)  # 字典列表储存的网络，模型信息
        self.net_info, self.module_list = create_modules(self.blocks)  # 网络的信息，模型（100多个sequential）
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]

    def forward(self, x, target=None):
        # 获取模型
        modules = self.blocks[1:]
        outputs = {}  # 保存feature_map
        out = []
        self.device = x.device
        for i, module in enumerate(modules):
            # i 索引modules和model_list的元素是相同的层
            module_type = (module["type"])
            # 如果时卷积层或者是上采样层,直接向前传播
            if module_type == "convolutional" or module_type == "upsample":
                if i != 87:
                    x = self.module_list[i](x)
                else:
                    if torch.tensor(self.module_list[i][0].weight).isnan().any():
                        ddd = 1
                    x = self.module_list[i](x)
            # 如果是concatenation 或者 回退到convolutional layer结束层
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                # 如果layer[0]索引大于0,让其和当前的层数做差
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                # 如果是回退到convolution set的输出
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                # 如果是concatenation连接（-1，index）
                else:
                    # 如果layer[1]索引大于0,让其和当前的层数做差，目的是为了求出需要连接的层和当前层的距离
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    # 上一层的输出
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    # 上一层+layer[2] 层的输出
                    x = torch.cat((map1, map2), 1)  # x.shape=(batch_size,channels,width,height)，所以是在dim=1的维度进行拼接
            # 残差连接
            elif module_type == "shortcut":
                from_ = int(module["from"])
                # （上一层）的输出加上（上一层-3）的输出
                x = outputs[i - 1] + outputs[i + from_]
            # Yolo检测层
            # 每次yolo层执行结束将会执行route(layer=-4)层
            elif module_type == 'yolo':
                # 获取特征图
                x = x.data  # （batch_size,3*(5+c),feature,feature)
                x = self.module_list[i][0](x)  # 输入一个元组（解码前，解码后）
                # 解码前：(bs,grid,grid,3,5+class) 解码后：(bs,grid,grid,3,5+class) or (bs*grid*grid*3, 5+class)
                out.append(x)  # 大，中，小
            outputs[i] = x
        if self.training:
            out = out[::-1]
            p, p_decoder = list(zip(*out))  # 小中大 (解码前，解码后)
            return p, p_decoder  # small, middle, large
        else:
            out = out[::-1]
            p, p_decoder = list(zip(*out))
            return p, torch.cat(p_decoder, 1)

    def load_weights(self, weight_file):
        # 权重初始化
        fp = open(weight_file, "rb")
        # 权重文件中前五个信息（int32）是
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        # 权重,类型为float32
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            # 只有卷积层和batch_normalization层才添加权重
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                    # 如果模型有bath_norm
                    # 一层卷积有（卷积层，batchNorm,leakyRelu）
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    # 如果有batch Norm Layer
                    # 数据顺序为bn_bias,bn_weight,bn_running_mean,bn_running_var,weight
                    bn = model[1]
                    # 获取batch norm layer的权重个数
                    num_bn_biases = bn.bias.numel()

                    # 加载偏执
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # 加载权重
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # 将加载的数据reshape成模型权重的形状
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # 将权重copy到batch norm layer 中
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    # 如果没有batch norm layer
                    # 注意weight文件中bias的数据在weight前
                    # 获取bias的数量
                    num_biases = conv.bias.numel()

                    # 获取偏置
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape偏置数据的形状，为模型参数的形状
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # 将偏置的数据加载进模型
                    conv.bias.data.copy_(conv_biases)
                # 最后，加载卷积层卷积核的权重,先获取卷积层权重的个数
                num_weights = conv.weight.numel()

                # 加载权重数据
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                # 修改形状
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_darknet_weights(self, path, cutoff=-1):
        """
        保存模型的权重
        """
        fp = open(path, "wb")
        self.header[3] = self.seen
        self.header.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.net_info[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # 如果有BN层先放置BN层的编制
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # 如果没有BN层则先放置卷积层的偏执
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # 再放置卷积层的权重
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()


