import numpy as np
import json
import pickle
from ir_preprocessing import load_json


def get_ans_dict(run_qrel_file):
    '''It works for run and qrel files'''
    ans = {}
    with open(run_qrel_file, 'rt') as f_in:
        for line in f_in:
            line = line.strip('\n')
            if line.split(' ')[3] == '1':
                qid = line.split(' ')[0]
                answer = line.split(' ')[2]
                ans[qid] = answer
    return ans

def load_predictions(retrieved_docs_file, gold_qrels_file):
    print('retrieved_docs_file: ', len(retrieved_docs_file))
    print('gold_qrels_file: ', len(gold_qrels_file))
    gold_dict = get_ans_dict(gold_qrels_file)
    pred_dict = get_ans_dict(retrieved_docs_file)
    
    golds = []
    preds = []
    
    for k in gold_dict.keys():
        golds.append(gold_dict[k])
        if k in pred_dict.keys():
            preds.append(pred_dict[k])
        else:
            preds.append('na')
    
    return [preds, golds]

def evaluate(predicted_answers, gold_answers):
    
    print('preds: ', len(predicted_answers))
    print('golds: ', len(gold_answers))
    
    preds = np.asarray(predicted_answers)
    targets = np.asarray(gold_answers)
    acc = sum(preds == targets) / float(len(preds))
    return round(float(acc),5)


