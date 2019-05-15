#coding: utf-8

import json
from utils.normalize import convert_to_squad, filter_tags

test_datasets = []
def prepare_test_squad(rank_file, preprocessed_file):
    with open(rank_file, 'r', encoding='utf-8') as f: 
        datas = json.load(f)
    with open(preprocessed_file, 'r', encoding='utf-8') as f: 
        for lidx, line in enumerate(f):
            data = {}
            sample = json.loads(line.strip())
            data['question_text'] = sample['question']
            data['qas_id'] = str(sample['question_id'])
            if data['qas_id '] in datas.keys():
                # data['doc'] = filter_tags(datas[data['qas_id']])[:500]
                ## 使用natualio方法预测
                paras = [filter_tags(para) for para in datas[data['qas_id']].split('##')]
                if len('##'.join(paras)) <= 500:
                    data['doc'] = '##'.join(paras)
                else:
                            
                    passage = '##'.join(paras[:2])
                    passage += '##'
                    for para in paras[2:]:
                        passage += para.split('。')[0]
                    data['doc'] = passage[:500]
            else:
                # print(data['qas_id'])
                data['doc'] = ''
            if 'answers' in sample.keys():
                data['orig_answer_text'] = sample['answers']
            test_datasets.append(data)
    return test_datasets


if __name__ == "__main__":
    zhidao_test_datasets = prepare_test_squad('./retriever/zhidao_test_rank_output.json',
                                                './data/test1_preprocessed/test1set/zhidao_test1.json')
    search_test_datasets = prepare_test_squad('./retriever/search_test_rank_output.json',
                                                './data/test1_preprocessed/test1set/search_test1.json')                                            
    squad_datas = convert_to_squad(zhidao_test_datasets+search_test_datasets)
    with open('./dureader_test.json', 'w', encoding='utf-8') as f:
        json.dump(squad_datas, f, ensure_ascii=False)
