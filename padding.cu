__global__ void padding(const float *A,int kw,int kh,int cusiz,float *C){
    int tx = threadIdx.x+blockIdx.x*cusiz;
    int ty = threadIdx.y+blockIdx.y*cusiz;

    int aw = blockDim.x*gridDim.x;
    int ah = blockDim.y*gridDim.y;
    int cw = aw+kw-1;
    int ch = ah+kh-1;

    int pw = (kw-1)/2;
    int ph = (kh-1)/2;

    int a_idx = tx + ty*aw;
    int c_idx = (tx+pw) + (ty+ph)*cw;

    C[c_idx]=A[a_idx];

    // 왼쪽 위
    if ( tx<=pw && ty<=ph ){
        C[tx + ty*cw] = A[0 + 0*aw];
    }
    // 오른쪽 위
    else if ( (aw-pw)<=tx && ty<=ph ){
        C[(2*pw+tx) + ty*cw] = A[(aw-1) + 0*aw];
    }
    // 왼쪽 아래
    else if ( tx<=pw && (ah-ph)<=ty ){
        C[tx + (2*ph+ty)*cw] = A[0 + (ah-1)*aw];
    }
    // 오른쪽 아래
    else if ( (aw-pw)<=tx && (ah-ph)<=ty ){
        C[(2*pw+tx) + (2*ph+ty)*cw] = A[(aw-1) + (ah-1)*aw];
    }
    // 위
    if ( ty<ph ){
        C[(pw+tx) + ty*cw] = A[tx + 0*aw];
    }
    // 아래
    else if ( ah-ph<=ty ){
        C[(pw+tx) + (2*ph+ty)*cw]=A[tx + (ah-1)*aw];
    }
    // 왼쪽
    else if ( tx<pw ){
        C[(tx) + (ty+ph)*cw] = A[0 + ty*aw];
    }
    // 오른쪽
    else if ( aw-pw<=tx ){
        C[(2*pw+tx) + (ty+ph)*cw] = A[(aw-1) + ty*aw];
    }
}