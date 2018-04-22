# -*- coding: UTF-8 -*-
import caffe
from caffe import layers as L, params as P, to_proto
import numpy as np
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
from copy import copy
import tools

sys.path.append('./DesignLayer')

from InitLandmark import InitLandmark
from SumOfSquaredLossLayer import SumOfSquaredLossLayer
# from TransformParamsLayer import TransformParamsLayer
# from AffineTransformLayer import AffineTransformLayer
# from LandmarkTranFormLayer import LandmarkTranFormLayer
# from GetHeatMapLayer import GetHeatMapLayer
# from Upscale2DLayer import Upscale2DLayer


def conv_relu(bottom, ks, nout, stride=1, pad=1, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant', value=0))
    # in_place是一种实际中为了减少内存数据的方法，默认使用较好
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    return fc, L.ReLU(fc, in_place=True)


class DeepAlignmentNetwork(object):
    def __init__(self, nStages):
        self.batchsize = 64

        self.nStages = nStages
        self.workdir = './'

    def getLabelsForDataset(self, imageServer):
        """生成当前imageServer的initLandmarks和gtLandmarks组合

        Args:
            imageServer: 欲处理的imageServer
        Return:
            一个nSamples*2*nLandmarks*2的数组，存储了nSamples*nLandmarks*2维度的initLandmarks和gtLandmarks
        """
        nSamples = imageServer.gtLandmarks.shape[0]
        nLandmarks = imageServer.gtLandmarks.shape[1]

        y = np.zeros((nSamples, 2, nLandmarks, 2), dtype=np.float32)
        y[:, 0] = imageServer.initLandmarks
        y[:, 1] = imageServer.gtLandmarks

        return y

    def loadData(self, trainSet, validationSet):
        """从trainSet,validationSet中读入imgServer的信息

        """
        self.nSamples = trainSet.gtLandmarks.shape[0]
        self.imageHeight = trainSet.imgSize[0]
        self.imageWidth = trainSet.imgSize[1]
        self.nChannels = trainSet.imgs.shape[1]

        # 对于train 为 1*3*height*width的矩阵
        # 对于valid 为 1*1*height*width的矩阵
        self.Xtrain = trainSet.imgs
        self.Xvalid = validationSet.imgs

        # Ytrain和Yvalid都是一个nSamples*2*nLandmarks*2的矩阵，存储了各自的initLandmarks和gtLandmarks
        self.Ytrain = self.getLabelsForDataset(trainSet)
        self.Yvalid = self.getLabelsForDataset(validationSet)

        # 测试index和验证index
        self.testIdxsTrainSet = range(len(self.Xvalid))
        self.testIdxsValidSet = range(len(self.Xvalid))

        # 由imageServer中的扰动函数和归一化函数得到的meanImg和stdDevImg
        self.meanImg = trainSet.meanImg
        self.stdDevImg = trainSet.stdDevImg
        self.initLandmarks = trainSet.initLandmarks[0]
        print("load data finished.")

    def createCNN(self, istrain):
        net = caffe.NetSpec()
        if istrain:
            net.s1_input, net.label = L.MemoryData(batch_size=self.batchsize, channels=self.nChannels, height=self.imageHeight, width=self.imageWidth, ntop=2)
        else:
            net.s1_input, net.label = L.MemoryData(batch_size=self.batchsize, channels=self.nChannels, height=self.imageHeight, width=self.imageWidth, ntop=2)

        # STAGE 1
        net.s1_conv1_1, net.s1_relu1_1 = conv_relu(net.s1_input, 3, 64)
        net.s1_batch1_1 = L.BatchNorm(net.s1_relu1_1)
        net.s1_conv1_2, net.s1_relu1_2 = conv_relu(net.s1_batch1_1, 3, 64)
        net.s1_batch1_2 = L.BatchNorm(net.s1_relu1_2)
        net.s1_pool1 = max_pool(net.s1_batch1_2, 2)

        net.s1_conv2_1, net.s1_relu2_1 = conv_relu(net.s1_pool1, 3, 128)
        net.s1_batch2_1 = L.BatchNorm(net.s1_relu2_1)
        net.s1_conv2_2, net.s1_relu2_2 = conv_relu(net.s1_batch2_1, 3, 128)
        net.s1_batch2_2 = L.BatchNorm(net.s1_relu2_2)
        net.s1_pool2 = max_pool(net.s1_batch2_2)

        net.s1_conv3_1, net.s1_relu3_1 = conv_relu(net.s1_pool2, 3, 256)
        net.s1_batch3_1 = L.BatchNorm(net.s1_relu3_1)
        net.s1_conv3_2, net.s1_relu3_2 = conv_relu(net.s1_batch3_1, 3, 256)
        net.s1_batch3_2 = L.BatchNorm(net.s1_relu3_2)
        net.s1_pool3 = max_pool(net.s1_batch3_2)

        net.s1_conv4_1, net.s1_relu4_1 = conv_relu(net.s1_pool3, 3, 512)
        net.s1_batch4_1 = L.BatchNorm(net.s1_relu4_1)
        net.s1_conv4_2, net.s1_relu4_2 = conv_relu(net.s1_batch4_1, 3, 512)
        net.s1_batch4_2 = L.BatchNorm(net.s1_relu4_2)
        net.s1_pool4 = max_pool(net.s1_batch4_2)
        if istrain:
            net.s1_fc1_dropout = L.Dropout(net.s1_pool4, dropout_ratio=0.5, in_place=True)
        else:
            net.s1_fc1_dropout = net.s1_pool4
        net.s1_fc1, net.s1_fc1_relu = fc_relu(net.s1_fc1_dropout, 256)
        net.s1_fc1_batch = L.BatchNorm(net.s1_fc1_relu)

        net.s1_output = L.InnerProduct(net.s1_fc1_batch, num_output=136,
                                bias_filler=dict(type='constant', value=0))
        net.s1_landmarks = L.Python(net.s1_output, module="InitLandmark",
                                        layer="InitLandmark",
                                        param_str=str(dict(initlandmarks=self.initLandmarks.tolist())))

        if self.nStages == 2:
            addDANStage(net)
            net.output = net.s2_landmarks
        else:
            net.output = net.s1_landmarks

        net.loss = L.Python(net.output, net.label, module="SumOfSquaredLossLayer",
                                        layer="SumOfSquaredLossLayer",
                                        loss_weight=1)
        return str(net.to_proto())

    def addDANStage(self, net):
        #CONNNECTION LAYERS OF PREVIOUS STAGE
        # TRANSFORM ESTIMATION
        net.s1_transform_params = L.Python(net.s1_landmarks, module="LandmarkTranFormLayer",
                                            layer="LandmarkTranFormLayer",
                                            param_str=str(dict(mean_shape=self.initlandmarks.tolist())))
        # IMAGE TRANSFORM
        net.s1_img_output = L.Python(net.s1_input, net.s1_transform_params,
                                        module="AffineTransformLayer",
                                        layer="AffineTransformLayer")
        # LANDMARK TRANSFORM
        net.s1_landmarks_affine = L.Python(net.s1_landmarks, net.s1_transform_params,
                                            module="LandmarkTransformLayer",
                                            layer="LandmarkTransformLayer")
        # HEATMAP GENERATION
        net.s1_img_heatmap = L.Python(net.s1_landmarks_affine, module="GetHeatMapLayer",
                                        layer="GetHeatMapLayer")
        # FEATURE GENERATION
        # 使用56*56而不是112*112的原因是，可以减少参数，因为两者最终表现没有太大差别
        net.s1_img_feature = fc_relu(net.s1_fc1_batch, 56*56)
        net.s1_img_feature = L.Reshape(net.s1_img_feature, shape=dict(dim=[-1, 1, 56, 56]))
        net.s1_img_feature = L.Python(net.s1_img_feature, module="Upscale2DLayer", layer="Upscale2DLayer", param_str=str(dict(scale_factor=2)))

        # CURRENT STAGE
        net.s2_input = L.Concat(net.s1_img_output, net.s1_img_heatmap, net.s1_img_feature)
        net.s2_input_batch = L.BatchNorm(net.s2_input)

        net.s2_conv1_1, net.s2_relu1_1 = conv_relu(net.s2_input_batch, 3, 64)
        net.s2_batch1_1 = L.BatchNorm(net.s2_relu1_1)
        net.s2_conv1_2, s2_net.relu1_2 = conv_relu(net.s2_batch1_1, 3, 64)
        net.s2_batch1_2 = L.BatchNorm(net.s2_relu1_2)
        net.s2_pool1 = max_pool(net.s2_batch1_2, 2)

        net.s2_conv2_1, net.s2_relu2_1 = conv_relu(net.s2_pool1, 3, 128)
        net.s2_batch2_1 = L.BatchNorm(net.s2_relu2_1)
        net.s2_conv2_2, net.s2_relu2_2 = conv_relu(net.s2_batch2_1, 3, 128)
        net.s2_batch2_2 = L.BatchNorm(net.s2_relu2_2)
        net.s2_pool2 = max_pool(net.s2_batch2_2)

        net.s2_conv3_1, net.s2_relu3_1 = conv_relu(net.s2_pool2, 3, 256)
        net.s2_batch3_1 = L.BatchNorm(net.s2_relu3_1)
        net.s2_conv3_2, net.s2_relu3_2 = conv_relu(net.s2_batch3_1, 3, 256)
        net.s2_batch3_2 = L.BatchNorm(net.s2_relu3_2)
        net.s2_pool3 = max_pool(net.s2_batch3_2)

        net.s2_conv4_1, net.s2_relu4_1 = conv_relu(net.s2_pool3, 3, 512)
        net.s2_batch4_1 = L.BatchNorm(net.s2_relu4_1)
        net.s2_conv4_2, net.s2_relu4_2 = conv_relu(net.s2_batch4_1, 3, 512)
        net.s2_batch4_2 = L.BatchNorm(net.s2_relu4_2)
        net.s2_pool4 = max_pool(net.s2_batch4_2)

        net.s2_pool4_flatten = L.Flatten(net.s2_pool4)
        if istrain:
            net.s2_fc1_dropout = L.Dropout(net.s2_pool4_flatten, dropout_ratio=0.5, in_place=True)
            # , include=dict(phase=caffe.TRAIN)
        else:
            net.s1_fc1_dropout = net.s2_pool4_flatten
        net.s2_fc1, net.s2_fc1_relu = fc_relu(net.s2_fc1_dropout, 256)
        net.s2_fc1_batch = L.BatchNorm(net.s2_fce_relu)

        net.s2_output = L.InnerProduct(net.s2_fc1_batch, num_output=136,
                                bias_filler=dict(type='constant', value=0))
        net.s2_landmarks = L.Eltwise(net.s2_output, net.s1_landmarks_affine)
        net.s2_landmarks = L.Python(net.s2_landmarks, net.s1_transform_params,
                                            module="LandmarkTranFormLayer",
                                            layer="LandmarkTranFormLayer")

    def get_prototxt(self, learning_rate = 0.001, num_epochs=100):
        self.solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(self.workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(self.workdir, "valnet.prototxt"))
        self.solverprototxt.sp['base_lr'] = str(learning_rate)
        self.solverprototxt.sp['test_interval'] = str(self.batchsize * 40)
        self.solverprototxt.write(osp.join(self.workdir, 'solver.prototxt'))
        # write train_val net.
        with open(osp.join(self.workdir, 'trainnet.prototxt'), 'w') as f:
            f.write(self.createCNN(True))
        with open(osp.join(self.workdir, 'valnet.prototxt'), 'w') as f:
            f.write(self.createCNN(False))
        print('get prototxt finished.')


    def train(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        solver = caffe.AdamSolver(osp.join(self.workdir, 'solver.prototxt'))
        print('Adam Solver finished------------------------')
        # 如果模型定义时有区分training和validation的不同phase，那么在solver中实际上是存在
        # 两个表示网络的成员变量：solver.net和solver.test_nets，注意，前者直接就是一个Net的对象，
        # 而后者是Net对象的列表，如果像GoogleNet那样，存在一个training和一个testing(validation
        # 而不是真正的testing，做测试的文件其实是deploy.prototxt)，那么应该通过solver.test_nets[0]
        # 来引用这个测试网络；另外，测试网络和训练网络应该是共享中间的特征网络层权重，
        # 只有那些标出include { phase: TRAIN }或者include { phase: TEST }的网络层有区分；
        # 训练数据train_X, train_Y必须是numpy中的float32浮点矩阵，
        # train_X维度是sample_num*channels*height*width，
        # train_Y是sample_num维度的label向量，
        # 这里sample_num必须是trainning输入batch_size的整数倍，
        # 为了方便，我在实际使用时每次迭代只在整个训练集中随机选取一个batch_size的图片数据放进去；
        solver.net.set_input_arrays(self.Xtrain, self.Ytrain)
        solver.test_nets[0].set_input_arrays(self.Xvalid, self.Yvalid)
        # solver.step(1)即迭代一次，包括了forward和backward，solver.iter标识了当前的迭代次数；
        solver.step(1)



        # data = np.random.randint(0, 256, (512, 3, 32, 32)).astype("float32")
        # net.blobs['data'].data = data
        # label = np.random.randint(0, 10, (512, 1, 1, 1)).astype("float32")
        # net.blobs['label'].data = label


        # data = np.random.randint(0, 256, (512, 3, 32, 32)).astype("float32")
        # net.blobs['data'].data[...] = data
        # label = np.random.randint(0, 10, (512, 1, 1, 1)).astype("float32")
        # net.blobs['label'].data[...] = label
