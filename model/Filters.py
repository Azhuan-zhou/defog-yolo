import math
import torch
import torchvision
import numpy as np
import cv2



class DIP(torch.nn.Module):
    def __init__(self, encoder_output_dim: int = 256):
        super(DIP, self).__init__()
        # Encoder Model
        self.encoder = torchvision.models.vgg16()
        # Changed 4096 --> 256 dimension
        self.encoder.classifier[6] = torch.nn.Linear(4096, encoder_output_dim, bias=True)

        # White-Balance Module
        self.wb_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 3, bias=True))

        # Gamma Module
        self.gamma_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 1, bias=True))

        # Sharpning Module
        self.gaussian_blur = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 5.0))
        self.sharpning_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 1, bias=True))

        # De-Fogging Module(尝试tmin也使用可学习的参数)
        self.defogging_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 1, bias=True))

        # Contrast Module
        self.contrast_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 1, bias=True))

        # Contrast Module
        self.tone_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 8, bias=True))

    def rgb2lum(self, img: torch.tensor):
        """
        获取LUM
        """
        img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :, :] + 0.06 * img[:, 2, :, :]
        return img

    def lerp(self, a: int, b: int, l: torch.tensor):
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b

    def dark_channel(self, x: torch.tensor):
        # 获取暗通道
        z = x.min(dim=1)[0].unsqueeze(1)
        return z

    def atmospheric_light(self, x: torch.tensor, dark: torch.tensor, top_k: int = 1000):
        """
        获得全局大气光
        """
        device = x.device
        h, w = x.shape[2], x.shape[3]
        imsz = h * w
        numpx = int(max(math.floor(imsz / top_k), 1))
        darkvec = dark.reshape(x.shape[0], imsz, 1)
        imvec = x.reshape(x.shape[0], 3, imsz).transpose(1, 2)
        indices = darkvec.argsort(1)
        indices = indices[:, imsz - numpx:imsz]
        atmMax = torch.zeros([x.shape[0], 1, 1]).to(device)
        for b in range(x.shape[0]):
            for ind in range(1, numpx):
                temp = imvec[b, indices[b, ind], :].max()
                if temp > atmMax[b, 0, 0]:
                    atmMax[b, 0, 0] = temp
        a = atmMax.squeeze(1).unsqueeze(2).unsqueeze(3)
        return a

    def blur(self, x: torch.tensor):
        """
        高斯滤波
        """
        return self.gaussian_blur(x)

    def defog(self, x: torch.tensor, latent_out: torch.tensor):
        """Defogging module is used for removing the fog from the image using ASM
        (Atmospheric Scattering Model).
        I(X) = (1-T(X)) * J(X) + T(X) * A(X)
        I(X) => image containing the fog.
        T(X) => Transmission map of the image.
        J(X) => True image Radiance.
        A(X) => Atmospheric scattering factor.

        Args:
            x (torch.tensor): Input image I(X)
            latent_out (torch.tensor): Feature representation from DIP Module.

        Returns:
            torch.tensor : Returns defogged image with true image radiance.
        """
        omega = self.defogging_module(latent_out).unsqueeze(2).unsqueeze(3)
        omega = self.tanh_range(omega, torch.tensor(0.1), torch.tensor(1.))
        dark_i = self.dark_channel(x)
        a = self.atmospheric_light(x, dark_i)
        i = x / a
        i = self.dark_channel(i)
        t = 1. - (omega * i)
        j = ((x - a) / (torch.maximum(t, torch.tensor(0.01)))) + a  # 这个0.01是否可以是可以学习的超参数
        return j

    def white_balance(self, x: torch.tensor, latent_out: torch.tensor):
        """ White balance of the image is predicted using latent output of an encoder.

        Args:
            x (torch.tensor): Input RGB image.
            latent_out (torch.tensor): Output from the last layer of an encoder.
        Returns:
            torch.tensor: returns White-Balanced image.
        """
        log_wb_range = 0.5
        wb = self.wb_module(latent_out)
        wb = torch.exp(self.tanh_range(wb, -log_wb_range, log_wb_range))

        color_scaling = 1. / (1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] +
                              0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1) * wb
        wb_out = wb.unsqueeze(2).unsqueeze(3) * x
        return wb_out

    def tanh01(self, x: torch.tensor):
        """
        激活函数
        """
        return torch.tanh(x) * 0.5 + 0.5

    def tanh_range(self, x: torch.tensor, left: float, right: float):
        """_summary_

        Args:
            x (torch.tensor): _description_
            left (float): _description_
            right (float): _description_

        Returns:
            _type_: _description_
        """
        return self.tanh01(x) * (right - left) + left

    def gamma_balance(self, x: torch.tensor, latent_out: torch.tensor):
        """
        白平衡
        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        log_gamma = torch.log(torch.tensor(2.5))
        gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(self.tanh_range(gamma, -log_gamma, log_gamma))
        g = torch.pow(torch.maximum(x, torch.tensor(1e-4)), gamma)
        return g

    def sharpning(self, x: torch.tensor, latent_out: torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        out_x = self.blur(x)
        y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        y = self.tanh_range(y, torch.tensor(0.1), torch.tensor(1.))
        s = x + (y * (x - out_x))
        return s

    def contrast(self, x: torch.tensor, latent_out: torch.tensor):
        """
        对比度
        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
        Returns:
            _type_: _description_
        """
        alpha = torch.tanh(self.contrast_module(latent_out))
        luminance = torch.minimum(torch.maximum(self.rgb2lum(x), torch.tensor(0.0)), torch.tensor(1.0)).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha)
        return contrast_image

    def tone(self, x: torch.tensor, latent_out: torch.tensor):
        """
        色调
        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        curve_steps = 8
        tone_curve = self.tone_module(latent_out).reshape(-1, 1, curve_steps)
        tone_curve = self.tanh_range(tone_curve, 0.5, 2)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        total_image = x * 0
        for i in range(curve_steps):
            total_image += torch.clamp(x - 1.0 * i / curve_steps, 0, 1.0 / curve_steps) \
                           * tone_curve[:, :, i].unsqueeze(2).unsqueeze(3)
        total_image *= curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        return total_image


    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # 图像encoder
        latent_out = torch.nn.functional.relu_(self.encoder(x))
        wb_out = self.white_balance(x, latent_out)
        gamma_out = self.gamma_balance(x, latent_out)
        sharpening_out = self.sharpning(x, latent_out)
        fog_out = self.defog(x, latent_out)
        contrast_out = self.contrast(x, latent_out)
        tone_out = self.tone(x, latent_out)
        return wb_out, gamma_out, fog_out, sharpening_out, contrast_out, tone_out


def save_image(image_tensor, out_x, b):
    image_tensor = (image_tensor * 255).squeeze(0).permute(1, 2, 0).detach().numpy()
    for i in range(len(out_x)):
        out = out_x[i]
        filter_name = b[i]
        out = (out * 255).squeeze(0).permute(1, 2, 0).detach().numpy()
        out = np.hstack((image_tensor, out))
        cv2.imwrite("check/processed/{}.png".format(filter_name), out)


if __name__ == '__main__':
    encoder_out_dim = 256
    # 图片路径
    image_path = 'check/image/img.png'
    # 使用PIL库打开图片
    image_pil = cv2.imread(image_path) / 255
    image_pil = torch.FloatTensor(image_pil).permute(2, 0, 1)
    # 定义转换操作，将图像转换为 PyTorch 的 Tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448, 448))
    ])
    image_tensor = transform(image_pil).unsqueeze(0)
    # 显示图像张量的形状
    print('Image Tensor Shape:', image_tensor.shape)
    model = DIP(encoder_output_dim=encoder_out_dim)
    print(model)
    out_x = model(image_tensor)
    print('output shape:', len(out_x))
    b = ['wb', 'gamma', 'defog', 'sharpen', 'contrast', 'tone']
    save_image(image_tensor, out_x, b)
