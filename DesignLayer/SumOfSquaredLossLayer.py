import caffe
import numpy as np

class SumOfSquaredLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
    def reshape(self,bottom,top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)
        pass
    def forward(self, bottom, top):
        self.eyedist = np.zeros((bottom[0].data.shape[0],1))
        self.diff[...] = bottom[0].data - bottom[1].data.reshape(bottom[1].data.shape[0],136)
        top[0].data[...] = self.landmarkErrorNorm(bottom[0].data, bottom[1].data)

    def backward(self, top, propagate_down, bottom):
        tempdiff = self.diff[...].reshape(bottom[0].data.shape[0],68,2) # 64*68*2
        tempdiff_square_sum = np.sqrt(np.sum(tempdiff**2, axis=2))  #64*68
        temp = np.zeros_like(tempdiff, dtype=np.float32)
        temp[:,:,0] = tempdiff_square_sum
        temp[:,:,1] = tempdiff_square_sum
        tempdiff = tempdiff / (temp * 68 * bottom[0].data.shape[0])
        for i in range(bottom[0].data.shape[0]):
            tempdiff[i] = tempdiff[i]/self.eyedist[i]
        bottom[0].diff[...] = tempdiff.reshape(bottom[0].data.shape[0], 136)

    def landmarkPairErrorNorm(self, predict, landmarks, index):
        gtLandmarks = landmarks.flatten().reshape(68, 2)

        predict = predict.reshape(68, 2)
        meanError = np.mean(np.sqrt(np.sum((predict - gtLandmarks)**2, axis=1)))
        self.eyedist[index] = np.linalg.norm((np.mean(gtLandmarks[36:42],axis=0) - np.mean(gtLandmarks[42:48], axis=0)), 2)
        loss = meanError / self.eyedist[index]
        return loss

    def landmarkErrorNorm(self, predict, landmarks):
        errors = np.zeros(landmarks.shape[0])
        for i in range(landmarks.shape[0]):
            errors[i] = self.landmarkPairErrorNorm(predict[i], landmarks[i], i)
        return np.mean(errors)
