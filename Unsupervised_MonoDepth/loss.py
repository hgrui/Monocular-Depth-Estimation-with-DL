import torch
import torch.nn as nn
import torch.nn.functional as F
# 损失函数
# 这里是本论文的精髓,通过构建的三个有效的损失函数,将图像深度估计问题刻画为无监督学习方案.
# 再夸一句,优秀的loss.

class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    # 构造尺度金字塔.
    # 可以在不同尺度的分辨率分别计算误差，从而形成总体误差.
    # 这里选择的尺度为4.
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            # interpolate 根据size对image进行上/下采样,此处为下采样.
            # bilinear 双线性插值法
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    # image : N*C*H*W
    # 计算图像水平方向的梯度
    def gradient_x(self, img):
        # 为保证输入输出尺寸一致,对原始image进行水平方向填0操作.
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx
    # 计算图像垂直方向的梯度
    def gradient_y(self, img):
        # 对原始image进行垂直方向填0操作
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        # img:原始左/右图像   disp:左/右视差图
        batch_size, _, height, width = img.size()

        # 生成初始化图像,这里不需要多通道数据,因此大小为 N*H*W
        # x_base,y_base分别初始化为水平方向,垂直方向从0到1的递增数据
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # 单通道的视差图 N*H*W
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        # flow_field, grid_sample  为什么要这样计算？没懂
        # backward mapping 的方法,需要利用双线性插值的方法对非像素点的位置进行估计.
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    # structural similarity index, 结构相似性,对局部图像变化敏感且可导.
    # 考虑局部图像明度(图像块的均值)l(x,y),局部图像对比度(图像块的方差)c(x,y),局部图像块的结构s(x,y)三部分
    # l(x,y) = (2*mu_x*mu_y+C1)/(mu_x^2+mu_y^2+C1)
    # c(x,y) = (2*sig_x*sig_y+C2)/(sig_x^2+sig_y^2+C2)
    # s(x,y) = (sig_xy+C3)/(sig_x*sig_y+C3)
    # SSIM(x,y) = l(x,y)*c(x,y)*s(x,y) 且 C3 = C2/2
    # 得到最终表达式
    # SSIM(x,y) = (2*mu_x*mu_y+C1)*(2*sig_xy+C2)/(mu_x^2+mu_y^2+C1)*(sig_x^2+sig_y^2+C2)
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    # 多尺度计算视差图的平滑度误差
    # 为保证平滑性，基于视差图梯度值构建误差,同时考虑图像深度的不连续性,加入原图像梯度作为边缘感知的权重
    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        # 选择每一行图像梯度绝对值的均值作为系数项
        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        # input: 网络预测的四尺度的视差图
        # target: 原左右图像

        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # 提取出左右视差估计图,尺度为4
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        # 根据右/左原图和左/右视差图生成左/右重建估计图
        # backward/inverse mapping, 所以重建左估计图时,需要左视差图和右原图.
        left_est = [self.generate_image_left(right_pyramid[i],
                    disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                     disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        # 在右(左)视差图基础上根据左(右)视差图处理,得到左(右)估计视差图
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        # 视差图的平滑度误差
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1
        # 基于视差重建图像和原图像的L1范数误差
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                    - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        # 基于视差重建图和原图像的SSIM误差
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                     left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                      right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        # 左右视差图一致性误差,根据左视差图可以将右视差图转换为左视差图
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        # 考虑Depth不连续的情况发生在边缘附近,这里保证深度图的平滑性与原图像的梯度一致.
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        # 对不同的损失函数考虑相应的权重,这里的话权重值均为1.
        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss
