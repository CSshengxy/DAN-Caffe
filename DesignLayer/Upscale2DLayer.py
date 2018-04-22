import caffe
import numpy as np
import scipy.ndimage

class Upscale2DLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_param(params)
        self.scale_factor = params['scale_factor']
    def reshape(self,bottom,top):
        """
        There is no need to reshape the data
        """
        top[0].reshape(bottom[0].shape[0], bottom[0].shape[1], bottom[0].shape[2] * 2, bottom[0].shape[3] * 3)

        pass
    def forward(self, bottom, top):
         temp = bottom[0].data[0][0]
         temp = scipy.ndimage.zoom(temp, 2, order=0)
         top[0].data[0][0] = temp

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """

        pass

    def check_params(params):
        """
        A utility function to check the parameters for the data layers.
        """

        required = ['scale_factor']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)
