#!/usr/bin/env sh

TOOLS=./build/tools
GLOG_logtostderr=0 GLOG_log_dir=DAN-caffe/Log/Log/ \
$TOOLS/caffe train \
--solver=DAN-caffe/proto/stage1_solver.prototxt
