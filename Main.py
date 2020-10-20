### CUDA로 Convolution Function 만들기
# kernal종류 여러개로 사진 처리
import numpy as np
import matplotlib.pyplot as plt

from CUDA_Kernal import convolution
from imagecall import imagecall
from select_kernel import ker_is

impath = "CUDAKernelStudy\\Calendar.png"
bias = np.array([0])

# 원본
Data = imagecall(impath).img
# plt.imshow(Data, cmap='gray')
# plt.title('Original')
# plt.imsave('CUDAKernelStudy\\Original.png', Data, cmap='gray')

# Motion blur (Updown)
Data = imagecall(impath).img
kernel = ker_is(mode='UpDown').kernel
for i in range(30):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\UpDown.png', Data, cmap='gray')

# Motion blur (LeftRight)
Data = imagecall(impath).img
kernel = ker_is(mode='LeftRight').kernel
for i in range(30):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\LeftRight.png', Data, cmap='gray')

# Gaussian blur (3x3)
Data = imagecall(impath).img
kernel = ker_is(mode='Gaussian').kernel
for i in range(5):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\Gaussian.png', Data, cmap='gray')

# Gaussian blur (5x5)
Data = imagecall(impath).img
kernel = ker_is(mode='Gaussian5').kernel
for i in range(5):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\Gaussian5.png', Data, cmap='gray')

# Bilateral blur
Data = imagecall(impath).img
kernel = ker_is(mode='Bilateral').kernel
for i in range(5):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\Bilateral.png', Data, cmap='gray')

# Laplacian blur
Data = imagecall(impath).img
kernel = ker_is(mode='Laplacian').kernel
for i in range(1):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\Laplacian.png', Data, cmap='gray')

# Sobelx blur
Data = imagecall(impath).img
kernel = ker_is(mode='Sobelx').kernel
for i in range(1):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\Sobelx.png', Data, cmap='gray')

# Sobely blur
Data = imagecall(impath).img
kernel = ker_is(mode='Sobely').kernel
for i in range(1):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
# plt.imsave('CUDAKernelStudy\\Sobely.png', Data, cmap='gray')