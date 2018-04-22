import caffe
import numpy as np

class SumOfSquaredLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
    def reshape(self,bottom,top):
        top[0].reshape(1)
        pass
    def forward(self, bottom, top):
        top[0].data = landmarkErrorNorm(bottom[0].data, bottom[1].data)

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

def landmarkPairErrorNorm(predict, landmarks):
    gtLandmarks = landmarks[1]

    predict = np.reshape(68, 2)
    meanError = np.mean(np.sqrt(np.sum((predict - gtlandmarks)**2, axis=1)))
    eyedist = (np.mean(gtLandmarks[36:42],axis=0) - np.mean(gtLandmarks[42:48], axis=0)).norm(2)
    loss = meanError / eyedist
    return loss

def landmarkErrorNorm(predict, landmarks):
    errors = np.zeros(landmarks.shape[0])
    for i in range(landmarks.shape[0]):
        errors[i] = landmarkPairErrorNorm(predict[i], landmarks[i])
    return np.mean(errors)
