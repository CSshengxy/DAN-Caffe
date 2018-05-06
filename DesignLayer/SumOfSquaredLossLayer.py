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
        # print(bottom[0].data.shape)
        # print(bottom[0].data)
        # print(bottom[1].data.shape)
        # print(bottom[1].data)
        top[0].data[...] = landmarkErrorNorm(bottom[0].data, bottom[1].data)
        print('forward once')

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

def landmarkPairErrorNorm(predict, landmarks):
    gtLandmarks = landmarks.flatten().reshape(68, 2)

    predict = predict.reshape(68, 2)
    # print('gtlandmarks:')
    # print(gtLandmarks)
    # print('predict')
    # print(predict)
    meanError = np.mean(np.sqrt(np.sum((predict - gtLandmarks)**2, axis=1)))
    eyedist = np.linalg.norm((np.mean(gtLandmarks[36:42],axis=0) - np.mean(gtLandmarks[42:48], axis=0)), 2)
    loss = meanError / eyedist
    return loss

def landmarkErrorNorm(predict, landmarks):
    errors = np.zeros(landmarks.shape[0])
    for i in range(landmarks.shape[0]):
        errors[i] = landmarkPairErrorNorm(predict[i], landmarks[i])
    return np.mean(errors)
