# BERT-Dureader
[lic2019](http://lic2019.ccf.org.cn/read)Dureader 2.0比赛BERT实现, 最终分数在test1测试集上为Rouge-L: 49.48, Bleu-4: 51.43.主要依赖为:
```
python==3.6
torch==0.4.1
pytorch_pretrained_bert==0.6.1
```

## 下载数据
从lic2019阅读理解赛道报名下载数据，这里只用到processed数据，之后将这些数据放入都data文件夹下，格式为

```
data:.
├─dev_preprocessed
│  └─devset
├─test1_preprocessed
│  └─test1set
├─test2_preprocessed
│  └─test2set
└─train_preprocessed
```

[官方Baseline仓库](https://github.com/baidu/DuReader)也提供了数据下载脚本，可以通过脚本下载之后将数据放为此格式

前往BERT [google官方仓库](https://github.com/google-research/bert)下载中文预训练模型(chinese_L-12_H-768_A_12)解压放在data目录下，使用pytorch_pretrained_bert将tensorflow模型转化为pytorch模型
```
export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A_12

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  $BERT_BASE_DIR/bert_model.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```

## 选择候选文档
在选择候选文档来输入到BERT进行预测时，参考了[Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)这篇文章的方法，使用BERT进行段落排序。但由于数据准备和时间稍显仓促，词方法检索出来的文档并未得到理想状态。之中可能还存在些许问题，尚有提升空间。首先从train_preprocessed中准备一个训练集，之后再通过BERT进行二分类，之后通过得到的相关度分数进行排序。但我在这里得到的排序结果并不好，所有对zhidao和search两个数据集采用了不同的后处理方法。
```bash
$ cd retriever
$ python prepare.py
$ python run_classifier.py --data_dir ./retriever_data/ --bert_model ../data/chinese_L-12_H-768_A-12/ --task_name MRPC --output_dir ./retriever_output --do_train --do_eval --train_batch_size 8
```
训练完成之后模型保存在文件夹 **./retriever_output** 中,接下来使用训练好的模型，对test测试集筛查选择相关的备选文档
```bash
$ python bert_rank.py --test_file ../data/test1_preprocessed/test1set/zhidao.test1.json --output_path ../zhidao_test_rank_output.json
$ python bert_rank.py --test_file ../data/test1_preprocessed/test1set/search.test1.json --output_path ../search_test_rank_output.json
```

## 使用BERT训练抽取模型
先将训练集进行预处理转化为squad格式，再通过run_dureader.py对bert模型进行微调,模型结果保存在reader_output中
```bash
$ cd reader
$ python prepare_squad.py
$ python run_dureader.py --bert_model ../data/chinese_L-12_H-768_A_12 --do_train --train_file ./dureader_train.json --train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 3.0 --max_seq_length 384 --doc_stride 128 --output_dir ./reader_output
```

## 结合检索和抽取对测试集进行预测
```
$ python prepare_test.py
$ python predict_dureader.py --bert_model ./data/chinese_L-12_H-768_A_12/ --bin_path ./reader/reader_output/pytorch_model.bin --predict_file ./dureader_test.json --output ./test1_output
```
