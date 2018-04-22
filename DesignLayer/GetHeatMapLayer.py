import caffe
import numpy as np
import itertools

IMGSIZE = 112
PATCH_SIZE = 16

class GetHeatMapLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Need one input to compute the heatmap")

        self.patch_size = PATCH_SIZE
        self.half_size = PATCH_SIZE / 2
        #itertools,product求笛卡尔积
        self.offsets = np.array(list(itertools.product(range(-self.half_size, self.half_size + 1), range(-self.half_size, self.half_size + 1))))


    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 1, IMGSIZE, IMGSIZE)

    def forward(self, bottom, top):
        self.landmarks = bottom[0].data
        for i in range(self.landmarks.shape[0]):
            top[0].data[i] = getHeatMap(self.landmarks[i])

    def backward(self, top, propagate_down, bottom):
        """
        No need for backward.
        """
    def getHeatMap(self, landmark):
        landmark = landmark.reshape((-1, 2))
        landmark = np.clip(landmark, self.half_size, IMGSIZE - 1 - self.half_size)
        imgs = zeros(landmark.shape[0], 1, IMGSIZE, IMGSIZE)
        for i in range(landmark.shape[0])
            imgs[i] = getHeatMapHelper(landmark[i])
        img = np.max(imgs, 0)
        return img

    def getHeatMapHelper(self, landmark):
        img = np.zeros(1, IMGSIZE, IMGSIZE)
        intLandmark = landmark.astype('int32')
        #location: 对于当前的landmark所计算的坐标范围
        locations = self.offsets + intLandmark
        dxdy = landmark - intLandmark

        offsetsSubPix = self.offsets - dxdy

        # offsetsSubPix*offsetsSubPix代表||（x,y)-landmark||^2
        # 最后得到的val是一个n*1的矩阵，对应着每个位置对于该landmark的亮度
        vals = 1 / (1 + np.sqrt(np.sum(offsetsSubPix * offsetsSubPix, axis=1) + 1e-6))
        img[0, locations[:, 1], locations[:, 0]] = vals
        return img
