# -*- coding: UTF-8 -*-
import caffe
from caffe import layers as L, params as P, to_proto
import os.path as osp
import tools


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
        self.batchsize = 26
        self.nStages = nStages
        self.protodir = './proto'

    def loadData(self, trainSet, validationSet):
        """从trainSet,validationSet中读入imgServer的信息

        """
        self.nSamples = trainSet.initLandmarks.shape[0]
        self.initLandmarks = trainSet.initLandmarks[0]
        print("load data finished.")

    def createCNN(self, istrain):
        net = caffe.NetSpec()
        if istrain:
            net.data = L.Data(source='./data/data_train_lmdb', backend=P.Data.LMDB, batch_size=self.batchsize, ntop=1)
            net.label = L.Data(source='./data/label_train_lmdb', backend=P.Data.LMDB, batch_size=self.batchsize, ntop=1)
        else:
            net.data = L.Data(source='./data/data_val_lmdb', backend=P.Data.LMDB, batch_size=self.batchsize, ntop=1)
            net.label = L.Data(source='./data/label_val_lmdb', backend=P.Data.LMDB, batch_size=self.batchsize, ntop=1)

        # STAGE 1
        net.s1_conv1_1, net.s1_relu1_1 = conv_relu(net.data, 3, 64)
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
        # if istrain:
        net.s1_fc1_dropout = L.Dropout(net.s1_pool4, dropout_ratio=0.5)
        # else:
            # net.s1_fc1_dropout = net.s1_pool4
        net.s1_fc1, net.s1_fc1_relu = fc_relu(net.s1_fc1_dropout, 256)
        net.s1_fc1_batch = L.BatchNorm(net.s1_fc1_relu)

        net.s1_output = L.InnerProduct(net.s1_fc1_batch, num_output=136,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant', value=0))
        net.s1_landmarks = L.Python(net.s1_output, module="InitLandmark",
                                        layer="InitLandmark",
                                        param_str=str(dict(initlandmarks=self.initLandmarks.tolist())))

        if self.nStages == 2:
            self.addDANStage(net, istrain)
            net.loss = L.Python(net.s2_landmarks, net.label, module="SumOfSquaredLossLayer",
                                            layer="SumOfSquaredLossLayer",
                                            loss_weight=1)
        else:
            net.loss = L.Python(net.s1_landmarks, net.label, module="SumOfSquaredLossLayer",
                                            layer="SumOfSquaredLossLayer",
                                            loss_weight=1)
        return str(net.to_proto())

    def addDANStage(self, net, istrain):
        #CONNNECTION LAYERS OF PREVIOUS STAGE
        # TRANSFORM ESTIMATION
        net.s1_transform_params = L.Python(net.s1_landmarks, module="TransformParamsLayer",
                                            layer="TransformParamsLayer",
                                            param_str=str(dict(mean_shape=self.initLandmarks.tolist())),
                                            propagate_down=0)
        # IMAGE TRANSFORM
        net.s1_img_output = L.Python(net.data, net.s1_transform_params,
                                        module="AffineTransformLayer",
                                        layer="AffineTransformLayer")
        # LANDMARK TRANSFORM
        net.s1_landmarks_affine = L.Python(net.s1_landmarks, net.s1_transform_params,
                                            module="LandmarkTranFormLayer",
                                            layer="LandmarkTranFormLayer",
                                            param_str=str(dict(inverse=False)),
                                            propagate_down=0)
        # HEATMAP GENERATION
        net.s1_img_heatmap = L.Python(net.s1_landmarks_affine, module="GetHeatMapLayer",
                                        layer="GetHeatMapLayer")
        # FEATURE GENERATION
        # 使用56*56而不是112*112的原因是，可以减少参数，因为两者最终表现没有太大差别
        net.s1_img_feature1 = L.InnerProduct(net.s1_fc1_batch, num_output=56*56,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant', value=0),
                                propagate_down=0)
        net.s1_img_feature1_relu = L.ReLU(net.s1_img_feature1, in_place=True)
        net.s1_img_feature2 = L.Reshape(net.s1_img_feature1_relu, reshape_param={'shape':{'dim':[-1,1,56,56]}})
        net.s1_img_feature3 = L.Python(net.s1_img_feature2, module="Upscale2DLayer", layer="Upscale2DLayer", param_str=str(dict(scale_factor=2)))

        # CURRENT STAGE
        net.s2_input = L.Concat(net.s1_img_output, net.s1_img_heatmap, net.s1_img_feature3)
        net.s2_input_batch = L.BatchNorm(net.s2_input)

        net.s2_conv1_1, net.s2_relu1_1 = conv_relu(net.s2_input_batch, 3, 64)
        net.s2_batch1_1 = L.BatchNorm(net.s2_relu1_1)
        net.s2_conv1_2, net.s2_relu1_2 = conv_relu(net.s2_batch1_1, 3, 64)
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
        # if istrain:
        net.s2_fc1_dropout = L.Dropout(net.s2_pool4_flatten, dropout_ratio=0.5)
        # else:
        #     net.s2_fc1_dropout = net.s2_pool4_flatten
        net.s2_fc1, net.s2_fc1_relu = fc_relu(net.s2_fc1_dropout, 256)
        net.s2_fc1_batch = L.BatchNorm(net.s2_fc1_relu)

        net.s2_output = L.InnerProduct(net.s2_fc1_batch, num_output=136,
                                weight_filler=dict(type='xavier'),
                                bias_filler=dict(type='constant', value=0))
        net.s2_landmarks = L.Eltwise(net.s2_output, net.s1_landmarks_affine)
        net.s2_landmarks = L.Python(net.s2_landmarks, net.s1_transform_params,
                                            module="LandmarkTranFormLayer",
                                            layer="LandmarkTranFormLayer",
                                            param_str=str(dict(inverse=True)))

    def get_prototxt(self, learning_rate = 0.001, num_epochs=100):
        if self.nStages == 1:
            trainnet_filename = "stage1_trainnet.prototxt"
            valnet_filename = "stage1_valnet.prototxt"
            solver_filename = "stage1_solver.prototxt"
        else:
            trainnet_filename = "stage2_trainnet.prototxt"
            valnet_filename = "stage2_valnet.prototxt"
            solver_filename = "stage2_solver.prototxt"
        self.solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(self.protodir, trainnet_filename), testnet_prototxt_path = osp.join(self.protodir, valnet_filename))
        self.solverprototxt.sp['base_lr'] = str(learning_rate)
        self.solverprototxt.sp['test_interval'] = str(40)
        self.solverprototxt.sp['test_iter'] = str(2)
        self.solverprototxt.sp['snapshot_prefix'] = '\"./result/snapshot\"'
        # self.solverprototxt.sp['test_iter'] = str(num_epochs*self.nSamples/self.batchsize)
        self.solverprototxt.write(osp.join(self.protodir, solver_filename))
        # write train_val net.
        with open(osp.join(self.protodir, trainnet_filename), 'w') as f:
            f.write(self.createCNN(True))
        with open(osp.join(self.protodir, valnet_filename), 'w') as f:
            f.write(self.createCNN(False))
        print('get prototxt finished.')
