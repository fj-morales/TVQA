import numpy as np
import json

def load_predictions(retrieved_docs_file, all_data_ids_equiv_file):
    preds = []
    golds = []
    
    with open(all_data_ids_equiv_file, 'rt') as ids_equiv_f:
        ids_equiv = json.load(ids_equiv_f)
    
    with open(retrieved_docs_file, 'rt') as r_file:
        
        ret_qs = {}
        for doc in r_file:
            qid_ori = doc.split(' ')[0]
            aux_id = doc.split(' ')[2]
            ret_qs[qid_ori] = aux_id
        
        for qid, gold in gold_answers_dict.items():
#             print('gold_qid: ', qid)
#             print('gold_value: ', gold)
            golds.append(int(gold))
    
            if str(qid) in ret_qs.keys():
                aux_id = ret_qs[qid]

                qid_aux = ids_equiv[aux_id].split('_')[0]
                qa_key = ids_equiv[aux_id]
    #             print('qid, aux, gold, : ', qid, aux_id, gold, qid_aux)
                if str(qid) == str(qid_aux):
                    pred = int(qa_key.split('_')[1].lstrip('a'))
                else:
                    pred = int(6) # Found wrong doc qid answer
            else:
                print('Qid not in retrieved: ', qid)
                
                pred = int(7) # Answer not found in index
    
    
#             try:
#                 aux_id = ret_qs[qid]
#                 print('qid, aux, gold, : ', qid, aux_id, gold)
#                 qid_aux = ids_equiv[aux_id].split('_')[0]
#                 if str(qid) == str(qid_aux):
#                     pred = int(qa_key.split('_')[1].lstrip('a'))
#                 else:
#                     pred = int(6) # Found wrong doc qid answer
#             except: 
#                 pred = int(7) # Answer not found in index
            preds.append(pred) 
                
#     print('len preds: ', len(preds))
#     print('len golds: ', len(gold_answers_dict))
    
    return [preds, golds]

def evaluate(predicted_answers, gold_answers):
    
#     print('preds: ', predicted_answers)
#     print('golds: ', gold_answers)
    
    preds = np.asarray(predicted_answers)
    targets = np.asarray(gold_answers)
    acc = sum(preds == targets) / float(len(preds))
    return round(float(acc),5)
