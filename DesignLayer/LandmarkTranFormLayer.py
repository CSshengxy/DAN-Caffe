import caffe
import numpy as np


class LandmarkTranFormLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.inverse = False;
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the transform landmark")
        params = eval(self.param_str)
        if 'inverse' in params.keys():
            self.inverse = params['inverse']

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape)

    def forward(self, bottom, top):
        self.landmarks = bottom[0].data
        self.transform = bottom[1].data
        self.transform_landmarks = zeros(self.landmarks.shape)
        self.transform_landmarks = self.transform_landmarks.astype('int32')
        for i in self.landmarks.shape[0]:
            self.transform_landmarks[i] = self.affine_transform_helper(self.landmarks[i],self.transform)
        top[0].data = self.transform_landmarks

    def backward(self, top, propagate_down, bottom):
        """
        No need for backward.
        """
    def affine_transform_helper(self, landmarks, transform):
        A = np.zeros((2, 2))
        A[0, 0] = transform[0]
        A[0, 1] = transform[1]
        A[1, 0] = transform[2]
        A[1, 1] = transform[3]
        t = transform[4:6]

        if self.inverse:
            A = np.linalg.inv(A)
            t = np.dot(-t, A)
        transformed_landmark = (T.dot(landmarks.reshape((-1, 2)), A) + t).flatten()
        return transformed_landmark
