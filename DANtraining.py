#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

import caffe
import os.path as osp
import sys
sys.path.append('./DesignLayer')


def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))
def check_accuracy(net, num_batches, batch_size = 10):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data.reshape(net.blobs['label'].data.shape[0],136)
        ests = net.blobs['s2_landmarks'].data
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.AdamSolver(osp.join('./solver.prototxt'))
print('Adam Solver finished------------------------')


solver.test_nets[0].share_with(solver.net)
for itt in range(101):
    solver.step(25)
    print('itt:{:3d}'.format((itt + 1) * 25), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 1)))
