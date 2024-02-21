# PyTorch 实现 XDoG 图像滤镜

## 简介

本项目是对2011年一个流行的JavaScript图像处理算法 [xdog-sketch](https://github.com/alexpeattie/xdog-sketch) 的PyTorch实现。原算法利用差分高斯（DoG）和扩展差分高斯（XDoG）滤波器进行图像增强和边缘检测，广泛应用于图像的艺术化处理和特征提取。

在保证效果一致的前提下，本版本针对PyTorch进行了优化，提高了处理效率和灵活性。通过调整算法参数，用户可以实现从细腻的素描效果到粗犷的艺术效果的转变，满足不同的图像处理需求。

## 参数调整指南

### Sigma 1 和 Sigma 2
- `Sigma 1` 和 `Sigma 2` 控制用于边缘检测的两个高斯函数的强度。较低的`Sigma 1`会产生更细腻的细节（模仿详细素描），而较高的`Sigma 1`则细节较少。
- `Sigma 2` 通常应该大于 `Sigma 1`。当`Sigma 2`远大于`Sigma 1`时，线条会更粗；反之，则更细。

### 阈值（Threshold）
- `Threshold` 定义了用于将图像二值化（转换为黑白）的亮度阈值，适用于非XDoG模式。较低的阈值意味着更多像素变为白色，产生更亮的图像和更细的线条；反之亦然。阈值非常敏感，因此图像很容易变得非常亮或非常暗。

### 锐化（Sharpen，p）
- `Sharpen (p)` 控制在使用XDoG模式时应用的锐化强度。较大的`p`值会夸大图像中存在的黑白边缘。

### Phi (φ)
- `Phi (φ)` 控制使用XDoG模式时应用的软阈值处理的陡峭度。较大的`φ`会导致图像中黑白过渡更加尖锐。

### Epsilon (ε)
- `Epsilon (ε)` 控制调整后的亮度值上升到哪个水平将变为白色。较高的`ε`会产生较暗的图像，黑色区域更多；反之亦然。较低的`ε`更接近于DoG模式的行为。

## 使用示例

```python
# 加载图像并应用预处理
pil_image = Image.open("imgs/model.jpg")
image_tensor = transform(pil_image).unsqueeze(0)

# 应用XDoG滤波器
xdog_result = XDoGFilter(image_tensor, sigma1=0.8, sigma2=2.5, phi=0.1, epsilon=20, sharpen=35)

# 保存处理后的图像
xdog_result_img = xdog_result.squeeze().detach().numpy().astype(np.uint8)
Image.fromarray(xdog_result_img).save('imgs/xdog_result.jpg')