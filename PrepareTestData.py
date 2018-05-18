from ImageServer import ImageServer
import numpy as np

commonSetImageDirs = ["../data/images/helen/testset/"]
commonSetBoundingBoxFiles = ["../data/boxesHelenTest.pkl"]

datasetDir = "./data/"

meanShape = np.load("./data/meanFaceShape.npz")["meanShape"]

commonSet = ImageServer(initialization='box')
commonSet.PrepareData(commonSetImageDirs, commonSetBoundingBoxFiles, meanShape, 0, 1000, False)
commonSet.LoadImages()
commonSet.CropResizeRotateAll()
commonSet.imgs = commonSet.imgs.astype(np.float32)
commonSet.Save(datasetDir, "commonSet.npz")
