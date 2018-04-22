#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

from ImageServer import ImageServer
import numpy as np

imageDirs = ["./data/images/helen/trainset/"]
boundingBoxFiles = ["./data/boxesHelenTrain.pkl"]

datasetDir = "./data/"

meanShape = np.load("./data/meanFaceShape.npz")["meanShape"]

trainSet = ImageServer(initialization='rect')
trainSet.PrepareData(imageDirs, None, meanShape, 100, 100000, True)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])
trainSet.NormalizeImages()
trainSet.Save(datasetDir)

validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)
