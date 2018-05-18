#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import cv2
workdir = './'
sys.path.append(workdir + 'DesignLayer')

caffe.set_mode_gpu()
caffe.set_device(0)

model_def = workdir + 'trainnet.prototxt'
model_weights = workdir + 'snapshot_iter_2500.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

image = caffe.io.load_image(workdir + 'data/images/helen/testset/30427236_1.jpg')
plt.imshow("原图",image)


inputImg, transform = self.CropResizeRotate(img, inputLandmarks)
inputImg = inputImg - self.meanImg
inputImg = inputImg / self.stdDevImg
