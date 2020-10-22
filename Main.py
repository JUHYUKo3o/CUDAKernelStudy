### CUDA로 Convolution Function 만들기
# kernal종류 여러개로 사진 처리
import numpy as np
import matplotlib.pyplot as plt

from CUDA_Kernal import convolution
from imagecall import imagecall
from select_kernel import ker_is

impath = "CUDAKernelStudy\\controla.png"
bias = np.array([0])

# 원본
Data = imagecall(impath).img
# RGB 분해
D1 = Data[:,:,0]    # R
D2 = Data[:,:,1]    # G
D3 = Data[:,:,2]    # B
# Choice Kernel
kernel = ker_is(mode='Sobely').kernel
# 2D Convolution 
D1 = convolution(D1,kernel,bias,mode='same').conv_2()
D2 = convolution(D2,kernel,bias,mode='same').conv_2()
D3 = convolution(D3,kernel,bias,mode='same').conv_2()
# RGB 합체
Data[:,:,0] = D1
Data[:,:,1] = D2
Data[:,:,2] = D3
# Image Check
plt.imshow(Data)
plt.show()

# 원본
Data = imagecall(impath).img
# RGB 분해
D1 = Data[:,:,0]    # R
D2 = Data[:,:,1]    # G
D3 = Data[:,:,2]    # B
# Choice Kernel
kernel = ker_is(mode='Gaussian5').kernel
for i in range(10):
    # 2D Convolution 
    D1 = convolution(D1,kernel,bias,mode='same').conv_2()
    D2 = convolution(D2,kernel,bias,mode='same').conv_2()
    D3 = convolution(D3,kernel,bias,mode='same').conv_2()
# RGB 합체
Data[:,:,0] = D1
Data[:,:,1] = D2
Data[:,:,2] = D3
# Image Check
plt.imshow(Data)
plt.show()
# plt.imsave('CUDAKernelStudy\\Kip.png', Data)

