from config import cfg
from model.YoloV3.yolov3 import load_model, Darknet
from model.Filters import DIP
import torch
from torch import nn


class cSE(nn.Module):
    def __init__(self, in_channels):
        # 通道注意力机制
        super().__init__()
        self.filters_name = ['wb', 'gamma', 'fog', 'sharpening', 'contrast', 'tone']
        self.filters = self.build_conv()

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # (bs, 6, 1, 1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)  # (bs, 3, 1, 1)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)  # (bs, 6, 1, 1)
        self.norm = nn.Softmax(dim=1)

    def build_conv(self):
        module_list = nn.ModuleList()
        for name in self.filters_name:
            modules = nn.Sequential()
            modules.add_module(
                f"conv_{name}",
                nn.Conv2d(
                    in_channels=3,
                    out_channels=1,
                    kernel_size=1,
                    bias=True,
                )
            )
            modules.add_module(f"leaky_{name}", nn.LeakyReLU(0.1))
            module_list.append(modules)
        return module_list

    def init_parameters(self):
        # Initialize parameters of Conv2d layers based on the specified initialization type
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, images):
        out = []
        for image, model in zip(images, self.filters):
            out.append(model(image))
        cat_out = torch.cat(out, dim=1)

        z = self.avgpool(cat_out)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2]
        z = self.Conv_Excitation(z)  # shape: [bs, c]
        z = self.norm(z)
        print('---------------------------')
        print('wb:', z[:, 0].squeeze().squeeze().squeeze())
        print('gamma:', z[:, 1].squeeze().squeeze().squeeze())
        print('fog:', z[:, 2].squeeze().squeeze().squeeze())
        print('sharpening:', z[:, 3].squeeze().squeeze().squeeze())
        print('contrast:', z[:, 4].squeeze().squeeze().squeeze())
        print('tone:', z[:, 5].squeeze().squeeze().squeeze())
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

    def __init__(self, cfg_file, hidden=256):
        super(YoloDIP, self).__init__()
        self.DIP = DIP(hidden)
        self.cSE = cSE(in_channels=6)
        self.YOLOV3 = Darknet(cfg_file)

    def init_weights(self, weights_path_yolov3=None):
        self.cSE.init_parameters()
        load_model(self.YOLOV3, weights_path_yolov3)
        print('model weights init successfully')

    def forward(self, x):
        if cfg.MODEL.filters:
            filter_images = self.DIP(x)
            x = self.cSE(filter_images)
        output = self.YOLOV3(x)
        return output, x


def load_model_defog(yolov3_cfg, device, weights_path=None):
    model = YoloDIP(cfg_file=yolov3_cfg).to(device)
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
        print('model load successfully')
    return model


if __name__ == '__main__':
    model = YoloDIP(cfg_file='./model-config/yolov3.cfg')
    print(model)
