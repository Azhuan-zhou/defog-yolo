from config import cfg
from model.YoloV3.yolov3 import load_model
from model.Filters import DIP
import torch
from torch import nn
from utils import util



class cSE(nn.Module):
    def __init__(self, in_channels):
        # 通道注意力机制
        super().__init__()
        self.conv_wb = nn.Conv2d(3, 1, kernel_size=1)
        self.conv_gamma = nn.Conv2d(3, 1, kernel_size=1)
        self.conv_fog = nn.Conv2d(3, 1, kernel_size=1)
        self.conv_sharpening = nn.Conv2d(3, 1, kernel_size=1)
        self.conv_contrast = nn.Conv2d(3, 1, kernel_size=1)
        self.conv_tone = nn.Conv2d(3, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, images):
        wb_out = self.conv_wb(images[0])
        gamma_out = self.conv_gamma(images[1])
        fog_out = self.conv_fog(images[2])
        sharpening_out = self.conv_sharpening(images[3])
        contrast_out = self.conv_contrast(images[4])
        tone_out = self.conv_tone(images[5])
        cat_out = torch.cat([wb_out, gamma_out, fog_out, sharpening_out, contrast_out, tone_out], dim=1)

        z = self.avgpool(cat_out)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2]
        z = self.Conv_Excitation(z)  # shape: [bs, c]
        z = self.norm(z)

        out = z[:, 0].reshape(-1, 1, 1, 1) * images[0] + \
              z[:, 1].reshape(-1, 1, 1, 1) * images[1] + \
              z[:, 2].reshape(-1, 1, 1, 1) * images[2] + \
              z[:, 3].reshape(-1, 1, 1, 1) * images[3] + \
              z[:, 4].reshape(-1, 1, 1, 1) * images[4] + \
              z[:, 5].reshape(-1, 1, 1, 1) * images[5]

        out = (out - out.min()) / (out.max() - out.min())
        return out


class YoloDIP(torch.nn.Module):
    """
    p: (small.middle,large) (bs,grid,grid,dx+dy+dw+dh+conf+class)
    p_decoder: (small.middle,large) (bs,grid,grid,x+y+w+h+conf+class) # x,y,w,h是原本尺寸下的预测框位置
    """

    def __init__(self, cfg_file, weights_path_yolov3=None, hidden=256):
        super(YoloDIP, self).__init__()
        self.dip = DIP(hidden)
        self.yolov3 = load_model(cfg_file, weights_path_yolov3)
        self.cSE = cSE(in_channels=6)

    def forward(self, x):
        filter_images = self.dip(x)
        cSE_x = self.cSE(filter_images)
        output = self.yolov3(cSE_x)
        return output


if __name__ == '__main__':
    net = YoloDIP(cfg_file='./model_config/yolov3.cfg').cuda()
    inp = torch.rand(3, 3, 416, 416).cuda()
    p, p_d = net(inp)
    label_sbbox = torch.rand(3, 52, 52, 3, 11).cuda()
    label_mbbox = torch.rand(3, 26, 26, 3, 11).cuda()
    label_lbbox = torch.rand(3, 13, 13, 3, 11).cuda()
    sbboxes = torch.rand(3, 150, 4).cuda()
    mbboxes = torch.rand(3, 150, 4).cuda()
    lbboxes = torch.rand(3, 150, 4).cuda()
    anchors = cfg.YOLO.ANCHORS
    anchors = util.get_anchors(anchors)
    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(anchors, cfg.YOLO.STRIDES)(p, p_d, label_sbbox,
                                                                                 label_mbbox,
                                                                                 label_lbbox, sbboxes,
                                                                                 mbboxes, lbboxes)
    print(loss)
