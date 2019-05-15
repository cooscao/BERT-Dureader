#coding: utf-8

import json


def filter_text(text):
    text = text.replace(u'\u3000', '').replace(u'\u001a_', '')
    text = text.replace(u'\u0005', '').split('##')[0]
    return text

if __name__ == "__main__":
    with open('../test1_output/predictions.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)
    with open('./result.json', 'w', encoding='utf-8') as f: 
        for k, v in datas.items():
            d = {}
            d['question_id'] = int(k.split('-')[0])
            d['question_type'] = ""
            d['answers'] = [filter_text(v).lstrip('。').strip()]
            d['yesno_answers'] = []
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
                