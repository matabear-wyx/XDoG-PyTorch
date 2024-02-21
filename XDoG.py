"""
作者: 吴宇熊

描述: 本脚本使用PyTorch实现了图像处理滤镜，包括高斯模糊、差分高斯（DoG）和扩展差分高斯（XDoG）滤波器，用于图像增强和边缘检测。
这是对 https://github.com/alexpeattie/xdog-sketch 项目中算法的PyTorch版本实现，旨在提供更加高效的张量操作和卷积处理能力。

创建日期: 2024-02-16
版本: 1.0
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn.functional import conv2d


# 定义高斯核生成函数
def gaussian_kernel(sigma):
    # 计算高斯核的尺寸，确保为奇数
    kernel_size = int(max(np.round(sigma * 3) * 2 + 1, 3))
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("The dimension must be an odd integer greater than or equal to 3")

    # 初始化高斯核数组
    kernel = np.zeros((kernel_size, kernel_size))
    two_sigma_square = 2 * sigma * sigma  # 计算高斯函数的分母部分
    centre = (kernel_size - 1) / 2  # 高斯核中心点坐标

    # 填充高斯核数值
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = np.sqrt((i - centre) ** 2 + (j - centre) ** 2)
            gaussian = (1 / np.sqrt(np.pi * two_sigma_square)) * np.exp(-distance ** 2 / two_sigma_square)
            kernel[i, j] = gaussian

    # 归一化高斯核，确保其和为1
    kernel /= np.sum(kernel)
    return kernel


# 定义应用卷积操作的函数
def apply_convolution(tensor, kernel):
    if isinstance(kernel, np.ndarray):
        kernel = torch.from_numpy(kernel).float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)

    # 计算padding值，以便输出尺寸与输入尺寸相同
    padding = (kernel.size(2) // 2, kernel.size(3) // 2)
    # 执行卷积操作
    result = conv2d(tensor.float(), kernel.float(), padding=padding, stride=1)
    return result


# 定义差分高斯（DoG）滤波函数
def DoGFilter(image_tensor, sigma1, sigma2, threshold):
    image_tensor *= 255.0
    # 生成两个高斯核
    kernel1 = gaussian_kernel(sigma1)
    kernel2 = gaussian_kernel(sigma2)
    # 应用卷积生成两个模糊图像
    imgA = apply_convolution(image_tensor, kernel1)
    imgB = apply_convolution(image_tensor, kernel2)
    # 计算两个模糊图像的差异
    diff = imgA - imgB
    # 使差异全为正值
    diffPositive = diff - torch.min(diff)
    # 归一化差异值
    relativeDiff = diffPositive / torch.max(diffPositive)
    # 应用阈值生成二值图像
    result = (relativeDiff > threshold).float() * 255.0
    return result


def soft_threshold(pixels, phi, epsilon):
    return torch.tanh(phi * (pixels - epsilon))


# 定义扩展差分高斯（XDoG）滤波函数
def XDoGFilter(image_tensor, sigma1, sigma2, phi, epsilon, sharpen=1):
    # 生成两个高斯核
    kernel1 = gaussian_kernel(sigma1)
    kernel2 = gaussian_kernel(sigma2)
    # 应用卷积生成两个模糊图像
    imgA = apply_convolution(image_tensor, kernel1)
    imgB = apply_convolution(image_tensor, kernel2)
    # 计算加权差异图像
    scaled_diff = (sharpen + 1) * imgA - sharpen * imgB
    # 应用加权差异增强图像细节
    sharpened = image_tensor * scaled_diff * 255.0
    # 生成基于epsilon的二值掩码
    mask = (image_tensor * scaled_diff - epsilon) > 0
    mask = mask.float()
    # 生成掩码的反转版本
    inverseMask = 1 - mask
    inverseMask = inverseMask.float()
    # 应用软阈值处理
    softThresholded = 1 + soft_threshold(sharpened, phi, epsilon)
    # 组合原始掩码和软阈值处理后的结果，调整最终图像的尺度
    result = mask + inverseMask * softThresholded
    resultScaled = result * 255.0 / torch.max(result)
    return resultScaled


# 图像预处理流程
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
])

# 使用示例
pil_image = Image.open("../imgs/model.jpg")
image_tensor = transform(pil_image).unsqueeze(0)

xdog_result = XDoGFilter(image_tensor, sigma1=0.8, sigma2=2.5, phi=0.1, epsilon=20, sharpen=35)

xdog_result_img = xdog_result.squeeze().detach().numpy()
xdog_result_img = (xdog_result_img).astype(np.uint8)
pil_img_xdog = Image.fromarray(xdog_result_img)
pil_img_xdog.save('imgs/xdog_result.jpg')

dog_result = DoGFilter(image_tensor, sigma1=0.8, sigma2=2.5, threshold=0.40)

dog_result_img = dog_result.squeeze().detach().numpy()
dog_result_img = (dog_result_img).astype(np.uint8)
pil_img_dog = Image.fromarray(dog_result_img)
pil_img_dog.save('imgs/dog_result.jpg')
