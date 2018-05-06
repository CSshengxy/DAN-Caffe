#!/usr/bin/env python

import sys
import h5py
caffe_root ='/home/parisiten/Projects/caffe/'
sys.path.insert(0,caffe_root+'python/DesignLayer')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

solver=caffe.SGDSolver('/home/parisiten/Projects/caffe/DAN-caffe/test_solver.prototxt')
solver.solve()
