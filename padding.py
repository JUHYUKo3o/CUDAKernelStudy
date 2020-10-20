# convolution 계산 함수

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools
import pycuda.driver as cuda

class padding():
    # CUDA Limit size
    cu_lim = 32
    def __init__(self,D,K,mode='vaild'):
        # D : Data, K = kernel,
        kw = int(K.shape[0]) # kernel width
        kh = int(K.shape[1]) # kernel height

        # size setting (padding)
        if mode == 'vaild':
            aw = D.shape[0]-kw+1
            ah = D.shape[1]-kh+1
            P = D
        elif mode == 'same':
            D = D.astype(np.float32)
            aw = int(D.shape[0])
            ah = int(D.shape[1])

            if (aw % self.cu_lim == 0):
                aw_n = int(aw/self.cu_lim)
            else : 
                aw_n = int(aw/self.cu_lim +1)
                
            if (ah % self.cu_lim == 0):
                ah_n = int(ah/self.cu_lim)
            else : 
                ah_n = int(ah/self.cu_lim +1)
            
            # result size
            P = np.zeros([aw+kw-1,ah+kh-1]).astype(np.float32)
            # Module
            mod = SourceModule(open("CUDAKernelStudy\\padding.cu", "r", encoding="utf-8").read())
            cu_pad = mod.get_function("padding")

            # allocate memory on device
            d_gpu = cuda.mem_alloc(D.nbytes)
            p_gpu = cuda.mem_alloc(P.nbytes)

            # memory copy (host to device)
            cuda.memcpy_htod(d_gpu, D)
            cuda.memcpy_htod(p_gpu, P)

            kw32 = np.int32(kw)
            kh32 = np.int32(kh)
            cusiz = np.int32(self.cu_lim)
            # padding by CUDA
            cu_pad(d_gpu,kw32,kh32,cusiz,p_gpu,block=(self.cu_lim,self.cu_lim,1),grid=(aw_n,ah_n,1))

            # memory copy (device to host)
            cuda.memcpy_dtoh(P, p_gpu)

            d_gpu.free()
            p_gpu.free()
        
        self.D = D
        self.P = P
        self.C = np.zeros([aw,ah])