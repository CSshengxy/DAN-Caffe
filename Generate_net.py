import sys
import os
import numpy as np
import caffe
from caffe import layers as L, params as P
from DeepAlignmentNetwork import DeepAlignmentNetwork
import tools
import os.path as osp

caffe_root = '../'  # this file is expected to be in {caffe_root}/DAN-caffe
sys.path.append(caffe_root + 'python/DesignLayer')

caffe.set_mode_gpu()
caffe.set_device(0)

workdir = './'
solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"),
                                    testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
solverprototxt.sp['base_lr'] = "0.05"
solverprototxt.write(osp.join(workdir, 'solver.prototxt'))

training = DeepAlignmentNetwork()


# write train net.
with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
    f.write(training.createCNN(2))

solver = caffe.AdamSolver(osp.join(workdir, 'solver.prototxt'))
