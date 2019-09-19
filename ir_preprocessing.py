#!/usr/bin/env python
# coding: utf-8

#############
# Imports
#############

import argparse
import datetime
import os
import json
import re

from ir_utils import *

# %load ir_baseline_bm25_rm3_indri.py


## Functions


def all_data_load(data_files_list, all_data_file):
    if not os.path.exists(all_data_file):
        all_data = []
        [all_data.extend(load_json(x)) for x in data_files_list]

        with open(all_data_file, 'wt') as all_f:
            json.dump(all_data, all_f, indent=4)
        return all_data
    else:
        return load_json(all_data_file)

def load_json(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)

def doc_to_trec(key, title):
    trec_answer = {}
    doc = '<DOC>\n' +             '<DOCNO>' + str(key) + '</DOCNO>\n' +             '<title>' + title + '</title>\n' +             '</DOC>\n'
    trec_answer[str(key)] = doc
    return trec_answer
    
def answers_to_trec(q_data, gold_answer_file):
    trec_answers = []
    seed = 0
    n_answers = 5
    ids_equivalences = {}
    gold_qrel = []
    for item in q_data:
        answer_keys = range(seed,seed+n_answers)
        answers = [item['a' + str(i)] for i in range(0,5)]
        for a_item in zip(answer_keys, answers):
            trec_answer = doc_to_trec(a_item[0], a_item[1])
            trec_answers.append(trec_answer)
            qa_key = str(item['qid']) + '_a' + str(a_item[0]%n_answers)
            ids_equivalences[str(a_item[0])] = qa_key
            
        seed += n_answers
        try:
            gold_qrel.append(str(a_item[0]) + ' 0 ' + str(item['qid']) + '_' + str(item['answer_idx']) + ' 1')
        except:
            pass # For all the original test questions without gold answer
#     print('equivs len: ', len(ids_equivalences))
    with open(all_data_ids_equiv_file, 'wt') as ids_e_f:
        json.dump(ids_equivalences, ids_e_f, indent=4)
                              
    print('Save gold file: ', gold_answer_file)
    with open(gold_answer_file, 'wt') as gold_file:
        for line in gold_qrel:
            gold_file.write(line + '\n')
    
    return [trec_answers, ids_equivalences]


def to_trecfile(docs, filename, compression = 'yes', query=False):
    print('Saving file: ', filename)
#     print(len(docs))
    
    if compression == 'yes':
        with gzip.open(filename,'wt') as f_out:
            if query == True:
                f_out.write('<parameters>\n')
#                 f_out.write('<index>\n' + index_loc + '\n</index>\n')
                
            for doc in docs:
                for key, value in doc.items():
                    f_out.write(value)
            if query == True:
                f_out.write('</parameters>\n')
    else:
        with open(filename,'wt') as f_out:
            if query == True:
                f_out.write('<parameters>\n')
#                 f_out.write('<index>\n' + index_loc + '\n</index>\n')
            for doc in docs:
                for key, value in doc.items():
                    f_out.write(value)
            if query == True:
                f_out.write('</parameters>\n')

def to_indri_queries(q_data, q_or_s):
    indri_queries = []
    inverted_ids = invert_ids(inverted_ids_file)

    for item in q_data:
#         print('id: ', item['qid'])
        q_a_ans_aux_ids = []
        for i in range(0,5):
            q_id_ans = str(item['qid']) + '_a' + str(i)
            q_a_ans_aux_ids.append(inverted_ids[q_id_ans])
            
            
#             print('equiv_dict: ', q_id_ans, inverted_ids[q_id_ans])
            
            
        if q_or_s == 'q':
            indri_queries.append(to_indri_query(item['qid'], item['q'], q_a_ans_aux_ids))
        if q_or_s == 's':
            indri_queries.append(to_indri_query(item['qid'], item['located_sub_text'], q_a_ans_aux_ids))
        
        
    return indri_queries

def invert_ids(inverted_ids_file):
    with open(all_data_ids_equiv_file, 'rt') as ids_equiv_f:
        ids_equiv = json.load(ids_equiv_f)
        inverted_id_equiv = {v: k for k, v in ids_equiv.iteritems()}
    return inverted_id_equiv

    
def q_data_to_trec_file(q_data, filename, q_or_s):
    indri_queries = to_indri_queries(q_data, q_or_s)
    to_trecfile(indri_queries, filename, compression = 'no', query = True)


def to_indri_query(q_id, query, q_a_ans_aux_ids):
    
    q_t = {}
#     q_text = '<query>\n<type>indri</type>\n' +          '<number>' + str(q_id) + '</number>\n' +         '<text>\n#combine( ' + remove_sc(query) + ')\n</text>\n'
    q_text = '<query>\n<type>indri</type>\n' +          '<number>' + str(q_id) + '</number>\n' +         '<text>\n' + remove_sc(query) + '\n</text>\n'
    for q_a_inv_id in q_a_ans_aux_ids:
        q_text = q_text + '<workingSetDocno>' + q_a_inv_id +'</workingSetDocno>\n'
    q_text = q_text + '</query>\n\n'
    q_t[q_id] = q_text
    return q_t                    
       
def remove_sc(text):
    text = re.sub(r'<eos>',' ',text) # My method
    text = re.sub(r'[^\w\s]',' ',text) # My method
    return text    

                

class fakeParser:
    def __init__(self):
        self.dataset = 'bioasq' 
#         self.build_index = True
        self.build_index = None
        self.fold = '1'
        self.gen_features = True
#         self.gen_features = None
        
    
    
if __name__ == "__main__":
    
    
#     ir_toolkit_location = '../../../indri-l2r/'

#     # create dataset files dir
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--pool_size', type=int, help='')
    
    args=parser.parse_args()
#     args = fakeParser()
    random_search = 'yes'
    start = datetime.datetime.now()
    print(start)

    train_file = './data/tvqa_train_processed.json'
    val_file = './data/tvqa_val_processed.json'
    test_file = './data/tvqa_test_public_processed.json' # Only for index
    
    test_questions_file = './data/tvqa_val_processed.json' # taking from the original validation file
    
    all_data_file = './data/tvqa_all.json'
    
    data_files_list = [train_file, val_file, test_file]
   
    q_data_all = all_data_load(data_files_list,all_data_file)
    
#     data_split = 'dev' # 'dev' / 'train'
    n_rand_iter = 1  # 5000
    pool_size = 1 # 20
    model_type = 's' # q(uestion) / s(ubtitle)
    
    workdir = './workdir/'
    
    
    
    all_data_ids_equiv_file = workdir + 'all_data_ids_equiv.json'
    inverted_ids_file = workdir + 'ids_equiv.json' 
    
    create_dir(workdir)
    
    hits = 1

    
    
    
#     #########
    
# #    q_data = q_data[0:100]
    
    #########
    
    
    to_index_input = workdir + 'index_inputs/'
    index_dir = workdir + 'index_dir/'
    all_query_files = workdir + 'query_files/'
    all_sub_files = workdir + 'sub_files/'
    all_retrieved_files = workdir + 'retrieved_files/'
    best_model_dir = workdir + 'best_ir_model/'
    
    create_dir(to_index_input)
    create_dir(index_dir)
    create_dir(all_query_files)
    create_dir(all_sub_files)
    create_dir(all_retrieved_files)
    create_dir(best_model_dir)
    
    # Convert answers to one index_inpu_trec_doc_file
    # Save the index numbering equivalences (newid = queryid_a0, queryid_a1, ...)
    
    
    data_splits = ['dev', 'train', 'test']

    index_input_file = to_index_input + 'index_input_file'
    gold_answer_qrels_file = workdir + 'gold_answer_qrels'
    
    [trec_answers, ids_equiv] = answers_to_trec(q_data_all, gold_answer_qrels_file) # Use all data instead the data split
    to_trecfile(trec_answers, index_input_file, compression = 'no')
    
    for data_split in data_splits:
        
        if data_split == 'test':
            input_file = test_questions_file
        else:
            input_file = './data/tvqa_new_' + data_split + '_processed.json'  # './data/tvqa_new_dev_processed.json'
        
        q_data = load_json(input_file)
        # Convert all questions / subtitles to one trec topics file 

        query_topics_file = all_query_files + 'query_indri_file_' + data_split
    #     q_data_to_trec_file(q_data, query_topics_file, q_or_s = 'q')

        subtitles_topics_file = all_query_files + 'subtitle_indri_query_file_' + data_split
        q_data_to_trec_file(q_data, subtitles_topics_file, q_or_s = 's')


