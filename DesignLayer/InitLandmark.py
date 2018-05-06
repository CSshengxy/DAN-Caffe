import caffe
import numpy as np

class InitLandmark(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_params(params)
        self.initlandmarks = np.array(params['initlandmarks']).flatten()
    def reshape(self,bottom,top):
        """
        There is no need to reshape the data
        """
        pass
        batch_size = bottom[0].shape[0]
        landmark_number = bottom[0].shape[1]
        top[0].reshape(batch_size, landmark_number)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[...] + self.initlandmarks


    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['initlandmarks']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
