#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

from ImageServer import ImageServer
from DeepAlignmentNetwork import DeepAlignmentNetwork

datasetDir = "./data/"

trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=38000_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
# "for validation we use a random subset of 100 images from the training set."(原文)
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")

training = DeepAlignmentNetwork(2)

training.loadData(trainSet, validationSet)

training.get_prototxt()
