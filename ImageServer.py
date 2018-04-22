# -*- coding: UTF-8 -*-
from scipy import ndimage
# numpy
# python没有提供真正的数组功能，大规模使用时list会变得很慢，而numpy提供了真正的数组功能，以及对数据快速处理的函数，numpy还有很多更高级的扩展依赖库，例如：
# Scipy, Matplotlib, Pandas
# 此外值得一提的是，Numpy内置函数处理数据的速度是C语言级别的。
import numpy as np
import utils
# cPicke 可以对python数据进行序列化，比如list,dict甚至一个类的对象，为了能够完整的保存并且可逆的恢复
import pickle
import glob
from os import path

class ImageServer(object):
    def __init__(self, imgSize=[112, 112], frameFraction=0.25, initialization='box',  color=False):
        # list 是python的内置数据类型，list中的数据类不一定相同的，而array中的类型必须全部相同。list相较于array增加了存储和消耗cpu
        # 以下四个以及最后的boundingBoxes在PrepareData中初始化
        self.origLandmarks = []
        self.filenames = []
        self.mirrors = []
        self.meanShape = np.array([])

        self.meanImg = np.array([])
        self.stdDevImg = np.array([])

        self.perturbations = []

        self.imgSize = imgSize
        self.frameFraction = frameFraction
        self.initialization = initialization
        self.color = color;

        self.boundingBoxes = []

    @staticmethod
    def Load(filename):
        """从.npy .npz文件中导入对应的（key,value）到当前Server, 同时对imgs成员变量增加一个维度.

        Args:
            filename: .npy .npz文件对应的路径名称
        """
        pass
        print('Loading data from .npz file......')
        # .npy .npz 文件,都是便于numpy读取的二进制文件
        # 可以将数组存储到.npy文件中，也可以将多个数组存储到.npz（即多个.npy的压缩文件中）
        # 存储时可指定存储数组名，便于读取，否则默认为arr_0,arr_1依次
        # 举例： re = np.savez("result.npz",a,b,sin_array=c)
        # re是一个 dictionary
        # 即可以通过re["arr_0"]访问原数组a,通过sin_array访问原数组c
        imageServer = ImageServer()
        arrays = np.load(filename)
        # update 把arrays的键值对更新到imageServer.__dict__中
        imageServer.__dict__.update(arrays)
        # len 对于矩阵代表他的维数
        # np.newaxis 增加一个新的维数
        # 如果img只有一张图片的话 增加一个维度
        if (len(imageServer.imgs.shape) == 3):
            imageServer.imgs = imageServer.imgs[:, np.newaxis]

        return imageServer

    def Save(self, datasetDir, filename=None):
        """存储当前ImageServer的(key, value)内容到filename中.

        Save函数的调用，保证了TestSetPreparation.py可以单独提前运行以处理数据，将临时数据存储到硬盘上。
        Args:
            datasetDir: 存储路径名
            filename: 欲存储的文件名，如为None默认格式为类似"dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz"
        """
        pass
        print('Saving data to .npz file......')
        if filename is None:
            filename = "dataset_nimgs={0}_perturbations={1}_size={2}".format(len(self.imgs), list(self.perturbations), self.imgSize)
            if self.color:
                filename += "_color={0}".format(self.color)
            filename += ".npz"

        arrays = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        np.savez(datasetDir + filename, **arrays)

    def PrepareData(self, imageDirs, boundingBoxFiles, meanShape, startIdx, nImgs, mirrorFlag):
        """获取训练集的boundingbox\imagelist\meanshape\landmark\mirrors存入相应成员变量中

        Args:
            imageDirs: 训练集所有图片所在路径
            boudingBoxFiles: boudingBoxFile文件路径 .pkl文件
            meanshape: 均值图像
            startIdx: 从训练集中选取此次训练的第一个图像的id
            nImgs: 此次训练的图像个数
            mirrorFlag: 还不懂是干什么用的
        """
        pass
        print('Preparing Data......')
        filenames = []
        landmarks = []
        boundingBoxes = []
        # 获取图片名列表、lanmark标注列表、boundingbox列表分别存入以上三个变量中
        for i in range(len(imageDirs)):
            print('Preparing Data......%dth dataset.....'%(i))
            filenamesInDir = glob.glob(imageDirs[i] + "*.jpg")
            filenamesInDir += glob.glob(imageDirs[i] + "*.png")
            # open() rb: 以二进制的格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式，返回一个file对象
            # .pkl文件就是对于每一个图片创建一个dict元素,key为图片名去掉后缀，value为bounding box的四个顶点的坐标
            # pickle.load() 最后返回一个dictionary
            if boundingBoxFiles is not None:
                boundingBoxDict = pickle.load(open(boundingBoxFiles[i], 'rb'), encoding='iso-8859-1')

            for j in range(len(filenamesInDir)):
                print('Preparing Data......%dth image.....'%(j))
                filenames.append(filenamesInDir[j])
                # 获取标注点文件名.pts，将landmark转换为x,y坐标的array数组录入到landmarks中 68*2
                ptsFilename = filenamesInDir[j][:-3] + "pts"
                landmarks.append(utils.loadFromPts(ptsFilename))

                if boundingBoxFiles is not None:
                    basename = path.basename(filenamesInDir[j])
                    boundingBoxes.append(boundingBoxDict[basename])


        filenames = filenames[startIdx : startIdx + nImgs]
        landmarks = landmarks[startIdx : startIdx + nImgs]
        boundingBoxes = boundingBoxes[startIdx : startIdx + nImgs]

        mirrorList = [False for i in range(nImgs)]
        if mirrorFlag:
            #最后得到的是一个2倍的images长度的mirrorlist,前面的mirrorlist值为false即不做镜像变换，后面的为true做镜像变换
            mirrorList = mirrorList + [True for i in range(nImgs)]
            # 拼接 axis default is 0，即按行拼接
            filenames = np.concatenate((filenames, filenames))
            # 按行拼接返回数组，作用同上
            landmarks = np.vstack((landmarks, landmarks))
            boundingBoxes = np.vstack((boundingBoxes, boundingBoxes))

        # prepare data 进行一系列的初始化
        self.origLandmarks = landmarks
        self.filenames = filenames
        # self.mirrors 代表是否被镜像了
        self.mirrors = mirrorList
        self.meanShape = meanShape
        self.boundingBoxes = boundingBoxes
        print('Preparing Data finished.')

    def LoadImages(self):
        """处理每一个训练的image.

        处理每一个image，来初始化初始化self.images(存入所有训练图像矩阵)，用mean_shape缩放得到的
        landmark初始化self.initLandmarks,用self.origLandmarks(通过读取.pts文件获得)来初始化self.gtLandmarks（gr-
        ound truth landmark）
        """
        pass
        print('Loading Images......')
        self.imgs = []
        self.initLandmarks = []
        self.gtLandmarks = []

        for i in range(len(self.filenames)):
            print('Loading ......%dth image.....'%(i))
            # ndimage.imread类似matlab的imread函数，对于RGB图像，返回height*width*3的矩阵， 对于灰度图返回height*width
            img = ndimage.imread(self.filenames[i])

            # 如果self.color为true保持rgb图不变灰度图叠加三次，结果为height*width*3 的img
            # 如果self.color为false保持灰度图不变rgb图rgb取平均，结果为height*width的img
            if self.color:
                if len(img.shape) == 2:
                    # 类似于np.concarenate
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2:
                    # 转换为灰度图像，将RGB三值取平均
                    img = np.mean(img, axis=2)
            img = img.astype(np.uint8)

            if self.mirrors[i]:
                # 对需要做特征变换的图像，特征点做一个镜像变换
                self.origLandmarks[i] = utils.mirrorShape(self.origLandmarks[i], img.shape)
                # 对图像做镜像变换，fliplr即数组左右翻转，
                img = np.fliplr(img)

            # 如果self.color为true，则调整矩阵维度先后，变为3*height*width
            # 如果self.color为false,则增加一个维度， 变为1*height*width
            if self.color:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[np.newaxis]

            groundTruth = self.origLandmarks[i]

            if self.initialization == 'rect':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape)
            elif self.initialization == 'similarity':
                bestFit = utils.bestFit(groundTruth, self.meanShape)
            elif self.initialization == 'box':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape, box=self.boundingBoxes[i])

            self.imgs.append(img)
            self.initLandmarks.append(bestFit)
            self.gtLandmarks.append(groundTruth)

        self.initLandmarks = np.array(self.initLandmarks)
        self.gtLandmarks = np.array(self.gtLandmarks)
        print('Loading Images finished')

    def GeneratePerturbations(self, nPerturbations, perturbations):
        print('GeneratePerturbations......')
        self.perturbations = perturbations
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)
        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        newImgs = []
        newGtLandmarks = []
        newInitLandmarks = []

        translationMultX, translationMultY, rotationStdDev, scaleStdDev = perturbations

        rotationStdDevRad = rotationStdDev * np.pi / 180
        translationStdDevX = translationMultX * (scaledMeanShape[:, 0].max() - scaledMeanShape[:, 0].min())
        translationStdDevY = translationMultY * (scaledMeanShape[:, 1].max() - scaledMeanShape[:, 1].min())

        for i in range(self.initLandmarks.shape[0]):
            print('Perturbations ......%dth image.....'%(i))
            for j in range(nPerturbations):
                tempInit = self.initLandmarks[i].copy()

                angle = np.random.normal(0, rotationStdDevRad)
                offset = [np.random.normal(0, translationStdDevX), np.random.normal(0, translationStdDevY)]
                scaling = np.random.normal(1, scaleStdDev)

                R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

                tempInit = tempInit + offset
                tempInit = (tempInit - tempInit.mean(axis=0)) * scaling + tempInit.mean(axis=0)
                tempInit = np.dot(R, (tempInit - tempInit.mean(axis=0)).T).T + tempInit.mean(axis=0)

                tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], tempInit, self.gtLandmarks[i])

                newImgs.append(tempImg)
                newInitLandmarks.append(tempInit)
                newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)
        print('GeneratePerturbations finished')

    def CropResizeRotateAll(self):
        print('GeneratePerturbations......')
        newImgs = []
        newGtLandmarks = []
        newInitLandmarks = []

        for i in range(self.initLandmarks.shape[0]):
            print('Perturbations ......%dth image.....'%(i))
            tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], self.initLandmarks[i], self.gtLandmarks[i])

            newImgs.append(tempImg)
            newInitLandmarks.append(tempInit)
            newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)
        print('GeneratePerturbations finished')

    def NormalizeImages(self, imageServer=None):
        self.imgs = self.imgs.astype(np.float32)

        if imageServer is None:
            self.meanImg = np.mean(self.imgs, axis=0)
        else:
            self.meanImg = imageServer.meanImg

        self.imgs = self.imgs - self.meanImg

        if imageServer is None:
            self.stdDevImg = np.std(self.imgs, axis=0)
        else:
            self.stdDevImg = imageServer.stdDevImg

        self.imgs = self.imgs / self.stdDevImg

        from matplotlib import pyplot as plt

        meanImg = self.meanImg - self.meanImg.min()
        meanImg = 255 * meanImg / meanImg.max()
        meanImg = meanImg.astype(np.uint8)
        if self.color:
            plt.imshow(np.transpose(meanImg, (1, 2, 0)))
        else:
            plt.imshow(meanImg[0], cmap=plt.cm.gray)
        plt.savefig("../meanImg.jpg")
        plt.clf()

        stdDevImg = self.stdDevImg - self.stdDevImg.min()
        stdDevImg = 255 * stdDevImg / stdDevImg.max()
        stdDevImg = stdDevImg.astype(np.uint8)
        if self.color:
            plt.imshow(np.transpose(stdDevImg, (1, 2, 0)))
        else:
            plt.imshow(stdDevImg[0], cmap=plt.cm.gray)
        plt.savefig("../stdDevImg.jpg")
        plt.clf()

    def CropResizeRotate(self, img, initShape, groundTruth):
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)
        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        destShape = scaledMeanShape.copy() - scaledMeanShape.mean(axis=0)
        offset = np.array(self.imgSize[::-1]) / 2
        destShape += offset

        A, t = utils.bestFit(destShape, initShape, True)

        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((img.shape[0], self.imgSize[0], self.imgSize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=self.imgSize)

        initShape = np.dot(initShape, A) + t

        groundTruth = np.dot(groundTruth, A) + t
        return outImg, initShape, groundTruth
