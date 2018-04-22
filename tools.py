# -*- coding: UTF-8 -*-
import numpy as np


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean=[128, 128, 128]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self, testnet_prototxt_path="testnet.prototxt",
                 trainnet_prototxt_path="trainnet.prototxt", debug=False):

        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.001'
        # TODO: 上一次更新的权重
        self.sp['momentum'] = '0.9'

        # speed:
        # 如果batch_size为100，则需要迭代100次才能将10000个数据全部执行完。因此test_iter设置为100， 称之为一个epoch
        self.sp['test_iter'] = '100'
        # 测试间隔，也就是每训练250次，才进行一次测试
        self.sp['test_interval'] = '250'

        # looks:
        # 每训练25次在屏幕上显示一次， 如果设置为0则不显示
        self.sp['display'] = '25'
        # snapshot: 快照， 设置训练多少次后被保存
        self.sp['snapshot'] = '2500'
        # 快照路径
        self.sp['snapshot_prefix'] = '"snapshot"'  # string within a string!

        # learning rate policy
        # lr_policy代表调整学习率base_lr的策略， fixed代表保持base_lr不变
        self.sp['lr_policy'] = '"fixed"'

        # important, but rare:
        self.sp['gamma'] = '0.1'
        # weight_decay 权重衰减项， 防止过拟合的一个参数
        self.sp['weight_decay'] = '0.0005'
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'
        self.sp['test_net'] = '"' + testnet_prototxt_path + '"'

        # pretty much never change these.
        # max_iter 最大迭代次数， 这个数如果设置太小， 会导致没有收敛。 设置太大又会导致震荡， 浪费时间
        self.sp['max_iter'] = '100000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25'  # this has to do with the display.
        self.sp['iter_size'] = '1'  # this is for accumulating gradients

        if (debug):
            self.sp['max_iter'] = '12'
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'
            self.sp['display'] = '1'

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
