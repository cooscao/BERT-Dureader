# coding: utf-8
# 通过训练好的BERT二分类网络，选择相关的一些文档作为候选文档。

import torch
import os
import json
import numpy as np
import re
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from pytorch_pretrained_bert import BertForSequenceClassification, BertConfig, BertTokenizer
from run_classifier import InputExample, InputFeatures, convert_examples_to_features

parser = argparse.ArgumentParser('Bert Ranker')
parser.add_argument()