#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import caffe
from ImageServer import ImageServer
from scipy import ndimage
import utils
import sys
sys.path.append('./DesignLayer')

caffe.set_mode_gpu()
caffe.set_device(0)
model_def = './proto/stage1_trainnet.prototxt'
model_weights = './result/snapshot_iter_40000.caffemodel'

net = caffe.Net(model_def,
                model_weights,
                caffe.TRAIN)

print(net.params["s1_conv2_1"][0].data)

net.forward()

print(net.blobs["s1_output"].data.shape)
