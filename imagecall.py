# 이미지 불러오는 함수
# 20.10.2주
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class imagecall():
    def __init__(self,impath):
        imgget = np.array(Image.open(impath).convert('L'))

        img01 = imgget.astype(np.float64).reshape((960,960))
        imgmin = img01-img01.min()
        img = imgmin/imgmin.max()
        self.img = img

        # self.img = self.img.reshape((self.img.shape[0],self.img.shape[1],1))
