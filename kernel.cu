__global__ void conv(const float *A,const float *K,const float *bias,int aw,int ah, int kw,int kh,float *C){
    // A : input data, K : kernal
    // aw : A's width, ah : A's height
    // kw : K's width, kh : K's height
    // block=(BLOCK_SIZE,BLOCK_SIZE,1)
    // grid =(cw/BLOCK_SIZE, ch/BLOCK_SIZE, Ker_SIZE)

    int tx = threadIdx.x+blockIdx.x*blockDim.x;
    int ty = threadIdx.y+blockIdx.y*blockDim.y;

    int cw = blockDim.x*gridDim.x;

    int i = blockIdx.z % kw;
    int j = blockIdx.z / kw;

    /* index  */
    int a_idx = tx + ty*aw + (i+j*aw);
    int k_idx = i + (j*kw);
    int c_idx = tx + (ty*cw);
    
    /* Convolution */
    C[c_idx] += A[a_idx]*K[k_idx]+bias[0]/(kw*kh);
}
