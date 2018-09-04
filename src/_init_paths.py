# --------------------------------------------------------
# xxx
# Copyright (c) 2016 xxx
# Licensed under The MIT License [see LICENSE for details]
# Written by Duino Du
# --------------------------------------------------------

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

caffe_path = '/home/deckyal/Downloads/caffe/caffe/build/install'#/home/duino/project/py-faster-rcnni/caffe_fast_rcnn'

# Add caffe to PYTHONPATH
caffe_path = osp.join(caffe_path, 'python')
add_path(caffe_path)
