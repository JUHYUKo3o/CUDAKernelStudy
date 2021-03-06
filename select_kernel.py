# Kernel class
import numpy as np

class ker_is():
    def __init__(self,mode='Gaussian'):
        if mode == 'Gaussian':
            ker = np.array([
                [1,2,1],
                [2,4,2],
                [1,2,1]])
            # ker = np.array([
            # [1,1,2,2,2,1,1],
            # [1,2,4,4,4,2,1],
            # [2,4,4,6,4,4,2],
            # [2,4,6,6,6,4,2],
            # [2,4,4,6,4,4,2],
            # [1,2,4,4,4,2,1],
            # [1,1,2,2,2,1,1]])
            self.kernel = ker/(ker.sum())
        elif mode == 'Gaussian5':
            ker = np.array([
                [ 1, 4, 7, 4, 1],
                [ 4,16,26,26, 4],
                [ 7,26,41,26, 7],
                [ 4,16,26,26, 4],
                [ 1, 4, 7, 4, 1]])
            # ker = np.array([
            # [1,1,1,1,1,1,1,2,1,1,1,1,1,1,1],
            # [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1],
            # [1,1,1,1,1,2,2,3,2,2,1,1,1,1,1],
            # [1,1,1,1,2,2,3,3,3,2,2,1,1,1,1],
            # [1,1,1,2,2,3,3,4,3,3,2,2,1,1,1],
            # [1,1,2,2,3,3,4,4,4,3,3,2,2,1,1],
            # [1,2,2,3,3,4,4,5,4,4,3,3,2,2,1],
            # [2,2,3,3,4,4,5,5,5,4,4,3,3,2,2],
            # [1,2,2,3,3,4,4,5,4,4,3,3,2,2,1],
            # [1,1,2,2,3,3,4,4,4,3,3,2,2,1,1],
            # [1,1,1,2,2,3,3,4,3,3,2,2,1,1,1],
            # [1,1,1,1,2,2,3,3,3,2,2,1,1,1,1],
            # [1,1,1,1,1,2,2,3,2,2,1,1,1,1,1],
            # [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1],
            # [1,1,1,1,1,1,1,2,1,1,1,1,1,1,1]])
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
        elif mode == 'Bilateral':
            ker = np.array([
            [1,2,0],
            [2,4,0],
            [1,2,0]])
            self.kernel = ker/(ker.sum())
        elif mode == 'Laplacian':
            ker = np.array([
            [0, 1,0],
            [1,-4,1],
            [0, 1,0]])
            self.kernel = ker
        elif mode == 'Sobelx':
            ker = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]])
            self.kernel = ker
        elif mode == 'Sobely':
            ker = np.array([
            [-1,-2,-1],
            [ 0, 0, 0],
            [ 1, 2, 1]])
            self.kernel = ker
