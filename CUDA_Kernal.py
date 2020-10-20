# convolution class
# 20.10.2주
import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools
import pycuda.driver as cuda

# 2D convolution function
class convolution:
    # CUDA Limit size
    cu_lim = 32

    def __init__(self,D,K,bias,mode='vaild'):
        # D : Data, K = kernel,
        # kernel, bias
        self.K = K.astype(np.float64)       # Kernel
        self.bias = bias.astype(np.float64) # bias
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

        # DATA (np.float64)
        self.D = D.astype(np.float64)   # Original DATA
        self.A = P.astype(np.float64)   # Padding DATA
        # convolution result
        self.C = np.zeros([aw,ah]).astype(np.float64)
    
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
        C = np.zeros(self.C.shape).astype(np.float32)
        # Module 불러오리
        mod = SourceModule(open("CUDAKernelStudy\\kernel.cu", "r", encoding="utf-8").read())
        conv = mod.get_function("conv")

        # allocate memory on device
        a_gpu = cuda.mem_alloc(A32.nbytes)
        k_gpu = cuda.mem_alloc(K32.nbytes)
        c_gpu = cuda.mem_alloc(C.nbytes)
        bias_gpu = cuda.mem_alloc(bias32.nbytes)
        
        # memory copy (host to device)
        cuda.memcpy_htod(a_gpu, A32)
        cuda.memcpy_htod(k_gpu, K32)
        cuda.memcpy_htod(c_gpu, C)
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
        cuda.memcpy_dtoh(C, c_gpu)

        a_gpu.free()
        k_gpu.free()
        c_gpu.free()
        bias_gpu.free()

        return (C).astype(np.float64)

