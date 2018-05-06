#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

from ImageServer import ImageServer
from DeepAlignmentNetwork import DeepAlignmentNetwork
import sys

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root + 'python')
sys.path.append(caffe_root + 'python/DesignLayer')

datasetDir = "./data/"

trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=38000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
# "for validation we use a random subset of 100 images from the training set."(原文)
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")


#The parameters to the FaceAlignmentTraining constructor are: number of stages and indices of stages that will be trained
#first stage training only
training = DeepAlignmentNetwork(2)
#second stage training only
#training = FaceAlignmentTraining(2, [1])

training.loadData(trainSet, validationSet)
trainSet = None
validationSet = None
# training.generate_lmdb()
training.get_prototxt()
#load previously saved moved
#training.loadNetwork("../DAN-Menpo.npz")

training.train()
