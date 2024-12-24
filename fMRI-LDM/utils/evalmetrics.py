import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy.ndimage as ndi


# 计算MAE（Mean Absolute Error）
def mae(input, output):
    return torch.mean(torch.abs(input - output))

# 计算MSE（Mean Squared Error）
def mse(input, output):
    return torch.mean((input - output) ** 2)

# 计算PSNR（Peak Signal-to-Noise Ratio）
def psnr_custom(input, output, max_val=1.0):
    mse_value = mse(input, output)
    if mse_value == 0:
        return float('inf')
    return 20 * torch.log10(torch.max(input) / torch.sqrt(mse_value))

# 计算SSIM（Structural Similarity Index）
def ssim_custom(input, output):
    input_np = input.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    # print(input_np.shape, output_np.shape)
    data_range = torch.max(input) - torch.min(input)
    ssim_value = ssim(input_np[0], output_np[0], data_range=data_range.item(), multichannel=True, win_size=11)
    return ssim_value

# 计算MSSSIM（Multi-Scale SSIM）
def gaussian_kernel(size, sigma):
    """生成高斯核，用于图像模糊"""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / kernel.sum()


def gaussian_kernel_3d(size, sigma):
    """
    生成三维高斯核
    :param size: 高斯核的大小 (应该是奇数)
    :param sigma: 高斯核的标准差
    :return: 三维高斯核
    """
    # 创建三维坐标网格
    ax = np.arange(-(size // 2), (size // 2) + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    # 计算三维高斯函数
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

    # 归一化使得核的和为1
    kernel /= np.sum(kernel)

    return kernel


def gaussian_filter(image, sigma, size=5):
    """
    对三维图像应用高斯模糊
    :param image: 三维图像
    :param sigma: 高斯核的标准差
    :param size: 高斯核的大小，默认是 5
    :return: 模糊后的图像
    """
    # 生成三维高斯核
    kernel = gaussian_kernel_3d(size, sigma)

    # 使用 ndi.convolve 对图像进行卷积
    return ndi.convolve(image, kernel)


def ssim_single_scale(img1, img2, window_size=11, sigma=1.5):
    """计算单尺度的 SSIM"""
    data_range = np.max(img1) - np.min(img1)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window = np.ones((window_size, window_size))  # 使用均值窗口
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)
    sigma1_sq = gaussian_filter(img1 ** 2, sigma) - mu1 ** 2
    sigma2_sq = gaussian_filter(img2 ** 2, sigma) - mu2 ** 2
    sigma12 = gaussian_filter(img1 * img2, sigma) - mu1 * mu2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return np.mean(ssim_map)


def msssim_custom(input, output, scales=[1, 2, 4], window_size=11, sigma=1.5):
    input_np = input.detach().cpu().numpy()[0]
    output_np = output.detach().cpu().numpy()[0]

    msssim = 1.0
    for scale in scales:
        # 对输入和输出图像进行下采样
        input_scaled = ndi.zoom(input_np, (1 / scale, 1 / scale, 1 / scale), order=1)
        output_scaled = ndi.zoom(output_np, (1 / scale, 1 / scale, 1 / scale), order=1)

        # 计算该尺度的 SSIM
        ssim_val = ssim_single_scale(input_scaled, output_scaled, window_size, sigma)

        # 累积结果
        msssim *= ssim_val

    # 返回多尺度 SSIM（通常取加权平均）
    return msssim ** (1.0 / len(scales))