#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

import caffe
import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
import sys
sys.path.append('./DesignLayer')

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.AdamSolver(osp.join('./proto/stage1_solver.prototxt'))
print('Adam Solver finished------------------------')
solver.solve()
