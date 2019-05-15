#coding: utf-8

import re

def filter_tags(htmlstr):
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
    re_br=re.compile('<br\s*?/?>')#处理换行
    re_h=re.compile('</?\w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('\n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    s = s.replace(' ', '').replace(u'\u3000', '').replace('&nbsp', '').replace('\n', '')
    # s=replaceCharEntity(s)#替换实体
    return s


def convert_to_squad(datas):
    data_sets = {}
    data_sets['version'] = '1.1'
    data_sets['data'] = []
    for data in datas:
        sample = {}
        sample['title'] = data['question_text']
        sample['id'] = data['qas_id']
        para_dict = {}
        para_dict['context'] = data['doc']
        para_dict['id'] = str(data['qas_id']) + '-1'
        qas_dict = {}
        qas_dict['id'] = para_dict['id'] + '-1'
        qas_dict['question'] = data['question_text']
        qas_dict['answers'] = [{"id":"1", "text": '', 
                               "answer_start": 0}]
        para_dict['qas'] = [qas_dict]
        sample['paragraphs'] = [para_dict]
        data_sets['data'].append(sample)
    return data_sets