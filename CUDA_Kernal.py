# convolution class
import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools
import pycuda.driver as cuda

from padding import padding

# 2D convolution function
class convolution:
    # CUDA Limit size
    cu_lim = 32

    def __init__(self,D,K,bias,mode='vaild'):
        self.K = K.astype(np.float64)       # Kernel
        self.bias = bias.astype(np.float64) # bias

        # padding
        pad = padding(D,K,mode)

        # DATA (np.float64)
        self.D = pad.D.astype(np.float64)   # Original DATA
        self.A = pad.P.astype(np.float64)   # Padding DATA
        # convolution result
        self.C = pad.C.astype(np.float64)
    
    def conv_2(self):
        # data형 정리
        A32 = self.A.astype(np.float32)
        K32 = self.K.astype(np.float32)
        bias32 = self.bias.astype(np.float32)
        # size 정리
        [aw,ah] = self.A.shape
        [kw,kh] = self.K.shape
        [cw,ch] = self.C.shape
    
        if (cw % self.cu_lim == 0):
            cw_n = int(cw/self.cu_lim)
        else : 
            cw_n = int(cw/self.cu_lim +1)
            
        if (ch % self.cu_lim == 0):
            ch_n = int(ch/self.cu_lim)
        else : 
            ch_n = int(ch/self.cu_lim +1)
        
        # 2D convolution function
        C32 = np.zeros(self.C.shape).astype(np.float32)
        # Module 불러오리
        mod = SourceModule(open("CUDAKernelStudy\\kernel.cu", "r", encoding="utf-8").read())
        conv = mod.get_function("conv")

        # allocate memory on device
        a_gpu = cuda.mem_alloc(A32.nbytes)
        k_gpu = cuda.mem_alloc(K32.nbytes)
        c_gpu = cuda.mem_alloc(C32.nbytes)
        bias_gpu = cuda.mem_alloc(bias32.nbytes)
        
        # memory copy (host to device)
        cuda.memcpy_htod(a_gpu, A32)
        cuda.memcpy_htod(k_gpu, K32)
        cuda.memcpy_htod(c_gpu, C32)
        cuda.memcpy_htod(bias_gpu, bias32)

        aw32 = np.int32(aw)
        ah32 = np.int32(ah)
        kw32 = np.int32(kw)
        kh32 = np.int32(kh)
        # convolution by CUDA
        conv(a_gpu,k_gpu,bias_gpu,
            aw32,ah32,kw32,kh32,
            c_gpu,
            block=(self.cu_lim,self.cu_lim,1),
            grid=(cw_n,ch_n,int(kw*kh)))
        
        # memory copy (device to host)
        cuda.memcpy_dtoh(C32, c_gpu)

        a_gpu.free()
        k_gpu.free()
        c_gpu.free()
        bias_gpu.free()
        C = (C32).astype(np.float64)
        
        C = (C+np.abs(C.min()))/(C+np.abs(C.min())).max()

        return C

