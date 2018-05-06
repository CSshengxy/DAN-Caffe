import caffe
import numpy as np

class AffineTransformLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the transform image")
    def reshape(self, bottom, top):
        imageHeight = bottom[0].shape[2]
        imageWidth = bottom[0].shape[3]
        top[0].reshape(bottom[0].shape[0], 1, imageHeight, imageWidth)

    def forward(self, bottom, top):
        self.images = bottom[0].data
        self.transform = bottom[1].data
        A = np.zeros((bottom[1].shape[0], 2, 2))
        A[:,0,0] = self.transform[:,0,0]
        A[:,0,1] = self.transform[:,0,1]
        A[:,1,0] = self.transform[:,0,2]
        A[:,1,1] = self.transform[:,0,3]
        t = self.transform[:,0,4:6]

        for i in range(self.images.shape[0]):
            A[i] = np.linalg.inv(A[i])
            t[i] = np.dot(-t[i], A[i])
            top[0].data[i] = self.affine_transform(self.images[i][0], A[i], t[i])

    def backward(self, top, propagate_down, bottom):
        """
        No need for backward.
        """
    def affine_transform(self, img, A, t):
        img_height = img.shape[0]
        img_width = img.shape[1]
        pixels = [(x, y) for x in range(img_height) for y in range(img_width)]
        pixels = np.array(pixels, dtype=np.float32)

        outPixels = np.dot(pixels, A) + t
        outPixels[:,0] = np.clip(outPixels[:,0], 0, img_height-2)
        outPixels[:,1] = np.clip(outPixels[:,1], 0, img_width-2)

        outPixelsMinMin = outPixels.astype('int32')
        outPixelsMaxMin = outPixelsMinMin + [1, 0]
        outPixelsMinMax = outPixelsMinMin + [0, 1]
        outPixelsMaxMax = outPixelsMinMin + [1, 1]

        dx = outPixels[:, 0] - outPixelsMinMin[:, 0]
        dy = outPixels[:, 1] - outPixelsMinMin[:, 1]

        pixels = pixels.astype('int32')

        outImg = np.zeros((1, img_height, img_width))
        outImg[0, pixels[:, 1], pixels[:, 0]] += (1 - dx) * (1 - dy) * img[outPixelsMinMin[:, 1], outPixelsMinMin[:, 0]]
        outImg[0, pixels[:, 1], pixels[:, 0]] += dx * (1 - dy) * img[outPixelsMaxMin[:, 1], outPixelsMaxMin[:, 0]]
        outImg[0, pixels[:, 1], pixels[:, 0]] += (1 - dx) * dy * img[outPixelsMinMax[:, 1], outPixelsMinMax[:, 0]]
        outImg[0, pixels[:, 1], pixels[:, 0]] += dx * dy * img[outPixelsMaxMax[:, 1], outPixelsMaxMax[:, 0]]

        return outImg
