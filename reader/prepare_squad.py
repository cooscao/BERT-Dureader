# coding: utf-8

# 去除掉很多无效数据
# for train set
import json
from tqdm import tqdm
from utils.normalize import convert_to_squad

ZHIDAO_FILE = '../data/train_preprocessed/zhidao.train.json'
SEARCH_FILE = '../data/train_preprocessed/search.train.json'
OUTPUT_FILE = './dureader_train.json'


def get_dataset(file_path):
    datasets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for lidx, line in enumerate(tqdm(f)):
            data = {}
            sample = json.loads(line.strip())
            if not len(sample['match_scores']):
                continue
            if sample['match_scores'][0] < 0.7:
                continue
            if not len(sample['answer_docs']):
                continue
            if sample['answer_docs'][0] >= len(sample['documents']):
                continue
            data['qas_id'] = sample['question_id']
            data['question_text'] = sample['question']
            doc = sample['documents'][int(
                sample['answer_docs'][0])]  # related_doc
            split_para = doc['segmented_paragraphs'][int(
                doc['most_related_para'])]
            ##
            else_para = ''
            for i in range(len(doc['segmented_paragraphs'])):
                if i != int(doc['most_related_para']):
                    else_para += doc['paragraphs'][i] + '##'
            para = ''.join(split_para)
            # 去除<>的代码
            if len(para) > 500:
                continue
            data['doc'] = (para + '##' + else_para)[:500]
            answer_span = sample['answer_spans']
            if not len(answer_span):
                continue
            data['orig_answer_text'] = ''.join(
                split_para[answer_span[0][0]:answer_span[0][1]+1])
            data['start_position'] = len(
                ''.join(split_para[:answer_span[0][0]]))
            data['end_position'] = data['start_position'] + \
                len(data['orig_answer_text'])
            datasets.append(data)
    return datasets


def main():
    train_datasets = get_dataset(ZHIDAO_FILE) + get_dataset(SEARCH_FILE)
    squad_data = convert_to_squad(train_datasets)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
