import numpy as np

class ker_is():
    def __init__(self,mode='Gaussian'):
        if mode == 'Gaussian7':
            ker = np.array([
            [1,1,2,2,2,1,1],
            [1,2,4,4,4,2,1],
            [2,4,4,6,4,4,2],
            [2,4,6,6,6,4,2],
            [2,4,4,6,4,4,2],
            [1,2,4,4,4,2,1],
            [1,1,2,2,2,1,1]])
            self.kernel = ker/(ker.sum())
        elif mode == 'Gaussian15':
            ker = np.array([
            [1,1,1,1,1,1,1,2,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1],
            [1,1,1,1,1,2,2,3,2,2,1,1,1,1,1],
            [1,1,1,1,2,2,3,3,3,2,2,1,1,1,1],
            [1,1,1,2,2,3,3,4,3,3,2,2,1,1,1],
            [1,1,2,2,3,3,4,4,4,3,3,2,2,1,1],
            [1,2,2,3,3,4,4,5,4,4,3,3,2,2,1],
            [2,2,3,3,4,4,5,5,5,4,4,3,3,2,2],
            [1,2,2,3,3,4,4,5,4,4,3,3,2,2,1],
            [1,1,2,2,3,3,4,4,4,3,3,2,2,1,1],
            [1,1,1,2,2,3,3,4,3,3,2,2,1,1,1],
            [1,1,1,1,2,2,3,3,3,2,2,1,1,1,1],
            [1,1,1,1,1,2,2,3,2,2,1,1,1,1,1],
            [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,2,1,1,1,1,1,1,1]])
            self.kernel = ker/(ker.sum())
        elif mode == 'UpDown':
            ker = np.array([
            [0,4,0],
            [0,1,0],
            [0,4,0]])
            # ker = np.array([
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0]])
            self.kernel = ker/(ker.sum())
        elif mode == 'LeftRight':
            ker = np.array([
            [0,0,0],
            [4,1,4],
            [0,0,0]])
            # ker = np.array([
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [3,3,3,3,3,2,2,2,2,2,1,1,1,2,2,2,2,2,3,3,3,3,3],
            # [3,3,3,3,3,2,2,2,2,2,1,1,1,2,2,2,2,2,3,3,3,3,3],
            # [3,3,3,3,3,2,2,2,2,2,1,1,1,2,2,2,2,2,3,3,3,3,3],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
            self.kernel = ker/(ker.sum())
        elif mode == 'Laplacian':
            ker = np.array([
            [0,-1,0],
            [-1,4,-1],
            [0,-1,0]])
            self.kernel = ker