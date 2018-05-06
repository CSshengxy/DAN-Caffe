#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

import caffe
import os.path as osp
import sys
sys.path.append('./DesignLayer')


caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.AdamSolver(osp.join('./solver.prototxt'))
print('Adam Solver finished------------------------')
solver.step(1)
