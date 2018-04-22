import caffe
import numpy as np

IMGSIZE = 112

class AffineTransformLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the transform image")
    def reshape(self, bottom, top):
        top[0].reshape(1,1,imageHeight,imageWidth)

    def forward(self, bottom, top):
        self.images = bottom[0].data
        self.transform = bottom[1].data
        A = np.zeros(2,2)
        A[0,0] = self.transform[0]
        A[0,1] = self.transform[1]
        A[1,0] = self.transform[2]
        A[1,1] = self.transform[3]
        t = self.transform[4:6]
        A = np.linalg.inv(A)
        t = np.dot(-t, A)
        self.output_image = np.zeros(self.images.shape)
        self.output_image = self.output_image.astype('float32')

        for i in range(self.images.shape[0]):
            self.output_image[i] = self.affine_transform(self.images[i],A,t)

        top[0].data = self.output_image

    def backward(self, top, propagate_down, bottom):
        """
        No need for backward.
        """
    def affine_transform(self, img, A, t):
        pixels = [(x, y) for x in range(IMGSIZE) for y in range(IMGSIZE)]
        pixels = np.array(pixels, dtype=np.float32)

        outPixels = np.dot(pixels, A) + t
        outPixels = np.clip(outPixels, 0, IMAGESIZE-1)

        outPixelsMinMin = outPixels.astype('int32')
        outPixelsMaxMin = outPixelsMinMin + [1, 0]
        outPixelsMinMax = outPixelsMinMin + [0, 1]
        outPixelsMaxMax = outPixelsMinMin + [1, 1]

        dx = outPixels[:, 0] - outPixelsMinMin[:, 0]
        dy = outPixels[:, 1] - outPixelsMinMin[:, 1]

        pixels = pixels.astype('int32')

        outImg = np.zeros((1, IMGSIZE, IMGSIZE))
        outImg[0, pixels[:, 1], pixels[:, 0]] += (1 - dx) * (1 - dy) * img[outPixelsMinMin[:, 1], outPixelsMinMin[:, 0]]
        outImg[0, pixels[:, 1], pixels[:, 0]] += dx * (1 - dy) * img[outPixelsMaxMin[:, 1], outPixelsMaxMin[:, 0]]
        outImg[0, pixels[:, 1], pixels[:, 0]] += (1 - dx) * dy * img[outPixelsMinMax[:, 1], outPixelsMinMax[:, 0]]
        outImg[0, pixels[:, 1], pixels[:, 0]] += dx * dy * img[outPixelsMaxMax[:, 1], outPixelsMaxMax[:, 0]]

        return outImage
