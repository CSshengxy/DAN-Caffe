import sys
import caffe
import numpy as np

class TransformParamsLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_params(params)
        self.mean_shape = params['mean_shape']
    def reshape(self,bottom,top):
        """
        每次forward之前都会调用一次, 用于将blobs reshape到需要的形状,
        不要在reshape方法里处理数据的数值, 因为reshape方法运行时,
        bottom传入的数据并不是forward来的数据, 确切的说, 都只是分配了空间并用0填充.
        通过top[index].reshape改变形状
        """
        top[0].reshape(bottom[0].shape[0], 1, 6);
        pass
    def forward(self, bottom, top):
        transformed_shape = bottom[0].data
        nSamples = bottom[0].shape[0]
        for i in range(nSamples):
            top[0].data[i] = self.bestFit(transformed_shape[i])


    def bestFit(self, transformed_shape):
        destination = np.array(self.mean_shape)
        source = transformed_shape.reshape((-1, 2))

        destMean = np.mean(destination, axis=0)
        srcMean = np.mean(source, axis=0)

        srcVec = (source - srcMean).flatten()
        destVec = (destination - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec, 2)**2
        b = 0
        for i in range(destination.shape[0]):
            b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i]
        b = b / np.linalg.norm(srcVec, 2)**2

        A = np.zeros((2, 2))
        A[0, 0] = a
        A[0, 1] = b
        A[1, 0] = -b
        A[1, 1] = a
        srcMean = np.dot(srcMean, A)

        return np.concatenate((A.flatten(), destMean - srcMean))



    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    required = ['mean_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
