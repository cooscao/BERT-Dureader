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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser('Bert Ranker')
parser.add_argument('--config_file', default='./retriever_output/bert_config.json',
                    type=str, help="the config file for bert")
parser.add_argument('--model_path', default='./retriever_output/pytorch_model.bin',
                    type=str, help="the trained model of bert")
parser.add_argument('--test_file', default='../data/test1_preprocessed/test1set/zhidao.test1.json',
                    type=str, required=True, help="the test_file.")
parser.add_argument('--vocab_file', default='../data/chinese_L-12_H-768_A-12/vocab.txt',
                    type=str, help="the vocab file for bert")
parser.add_argument('--max_length', default=256,
                    type=int, help="the max length of a sentence")
parser.add_argument('--output_path', default='./rank_output.json',
                    type=str, required=True, help="the output file path")
args = parser.parse_args()


label_list = ["0", "1"]
device = torch.device('cuda')
tokenizer = BertTokenizer(args.vocab_file)
config = BertConfig(args.config_file)
model = BertForSequenceClassification(config, num_labels=2)
model.load_state_dict(torch.load(args.model_path, map_location=device))


def get_datas(path):
    datasets = []
    with open(path, 'r', encoding='utf-8') as reader:
        for lidx, line in enumerate(tqdm(reader)):
            dataset = []
            sample = json.loads(line.strip())
            question = sample['question']
            for i, doc in enumerate(sample['documents']):
                q_id = str(sample['question_id'])
                for para in doc['paragraphs']:
                    dataset.append([q_id, question, para])            
                datasets.append(dataset)
    return datasets


datasets = get_datas(args.test_file)
output_dict = {}
model.to(device)
print('Predicting by Bert....')
predicts = []
d = defaultdict(str)
for dataset in tqdm(datasets):
    if not len(dataset):
        # output_dict[pid] = ''
        continue
    examples = []
    for i, data in enumerate(dataset):
        if i < 64:
            examples.append(InputExample(i, data[1], data[2], '0'))
    eval_features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
   #  print(all_input_ids.size())
    input_ids = all_input_ids.to(device)
    input_mask = all_input_mask.to(device)
    segment_ids = all_segment_ids.to(device)
    label_ids = all_label_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)
            
    output = F.softmax(logits, dim=1)
    pid = dataset[0][0]
    max_index = torch.argmax(output[:, 1], dim=0).cpu().item()
    d[pid] += dataset[max_index][2] + '##'

with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False)
print('Done')
