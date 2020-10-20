### CUDA로 Convolution Function 만들기
## 10월 2주차
# kernal종류 여러개로 사진 처리
import numpy as np
import matplotlib.pyplot as plt

from CUDA_Kernal import convolution
from imagecall import imagecall
from select_kernel import ker_is

impath = "CUDAKernelStudy\\Ryan.png"
bias = np.array([0])

# 원본
Data = imagecall(impath).img
plt.subplot(2, 3, 1)
plt.imshow(Data, cmap='gray')
plt.title('Ryan')

# Gaussian blur (7x7)
Data = imagecall(impath).img
kernel = ker_is(mode='Gaussian7').kernel
for i in range(5):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
plt.subplot(2, 3, 2)
plt.imshow(Data, cmap='gray')
plt.title('Gaussian7')

# Gaussian blur (15x15)
Data = imagecall(impath).img
kernel = ker_is(mode='Gaussian15').kernel
for i in range(5):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
plt.subplot(2, 3, 3)
plt.imshow(Data, cmap='gray')
plt.title('Gaussian15')

# Motion blur (Updown)
Data = imagecall(impath).img
kernel = ker_is(mode='UpDown').kernel
for i in range(30):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
plt.subplot(2, 3, 4)
plt.imshow(Data, cmap='gray')
plt.title('UpDown')

# Motion blur (LeftRight)
Data = imagecall(impath).img
kernel = ker_is(mode='LeftRight').kernel
for i in range(30):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
plt.subplot(2, 3, 5)
plt.imshow(Data, cmap='gray')
plt.title('LeftRight')

# Motion blur (High-passfilter)
Data = imagecall(impath).img
kernel = ker_is(mode='Laplacian').kernel
for i in range(1):
    Data = convolution(Data,kernel,bias,mode='same').conv_2()
plt.subplot(2, 3, 6)
plt.imshow(Data, cmap='gray')
plt.title('Laplacian')

plt.show()
