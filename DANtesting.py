import numpy as np
from matplotlib import pyplot as plt
import caffe
from ImageServer import ImageServer

Channels = 1
imageHeight = 112
imageWidth = 112

caffe.set_mode_gpu()
caffe.set_device(0)
model_def = './trainnet.prototxt'
model_weights = './snapshot_iter_2500.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

testSet = ImageServer.Load("commonSet.npz")
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=38000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")

errors = landmarkError(trainSet, testSet, normalization='centers', showResults=True)

print("The test errors:")
print(errors)



def CropResizeRotate(self, img, inputShape, initLandmarks):
    A, t = utils.bestFit(initLandmarks, inputShape, True)

    A2 = np.linalg.inv(A)
    t2 = np.dot(-t, A2)

    outImg = np.zeros((Channels, imageHeight, imageWidth), dtype=np.float32)
    for i in range(img.shape[0]):
        outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=(imageHeight, imageWidth))

    return outImg, [A, t]


def landmarkError(ReferServer, ImageServer, normalization='centers', showResults=True):
    errors = []
    nImgs = len(imageServer.imgs)
    for i in range(nImgs):
        initLandmarks = imageServer.initLandmarks[i]
        gtLandmarks = imageServer.gtLandmarks[i]
        img = imageServer.imgs[i]

        if img.shape[0] > 1:
            img = np.mean(img, axis=0)[np.newaxis]

        resLandmarks = initLandmarks
        inputImg, transform = CropResizeRotate(img, resLandmarks, ReferServer.initLandmarks[0])
        inputImg = inputImg - ReferServer.meanImg
        inputImg = inputImg / ReferServer.stdDevImg

        # TODO: 读入网络结构，根据当前图片获取输出
        net.blobs['data'].data[...] = inputImg
        net.blobs['label'].data[...] = gtLandmarks
        outputblob = net.forward()
        output = output['s2_landmarks'][0]

        landmarks = output.reshape((-1, 2))
        resLandmarks = np.dot(landmarks - transform[1], np.linalg.inv(transform[0]))

        if normalization == 'centers':
            normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        elif normalization == 'corners':
            normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        elif normalization == 'diagonal':
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist = np.sqrt(width ** 2 + height ** 2)

        error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks)**2,axis=1))) / normDist
        errors.append(error)

        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()

    avgError = np.mean(errors)
    print "Average error: {0}".format(avgError)

    return errors
