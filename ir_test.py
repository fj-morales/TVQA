import sys
import subprocess
from eval_utils import *

# Functions
def generate_run_file(pre_run_file, run_file):
    
    with open(pre_run_file, 'rt') as input_f:
        pre_run = input_f.readlines()
#         print(type(pre_run))
    with open(run_file, 'wt') as out_f:
        for line in pre_run:
            out_f.write(line.replace('docid=','').replace('indri', 'lambdaMART'))


def gen_run_file(ranklib_location, normalization, save_model_file, test_data_file, run_file):
# Works also for testing
    ranker_command = ['java', '-jar', ranklib_location + 'RankLib-2.12.jar']
    pre_run_file = run_file.replace('run_', 'pre_run_', 1)
    toolkit_parameters = [
        *ranker_command, # * to unpack list elements
        '-load',
        save_model_file,
        *normalization,
        '-rank',
        test_data_file,
        '-indri',
        pre_run_file     
        ]             

    proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    (out, err)= proc.communicate()
    #         print(out.decode('utf-8').splitlines())
    #         print(out)
    #         print(err)

    generate_run_file(pre_run_file, run_file)

    
    
def test_model(workdir, ranklib_location, normalization, res):
    
    
    fold = '' # For compatibility
    fold_dir = workdir # For compatibility
    all_retrieved_files = workdir + 'retrieved_files/'
    gen_features_dir = workdir + 'gen_features_dir/'

#     test_questions_file = './data/tvqa_val_processed.json' # val here is test for our case
#     test_ids_equiv_file = workdir + 'test_ids_equiv.json'
    gold_answer_qrels_file = workdir + 'gold_answer_qrels_' + 'test'

    ## Evaluate on test

    

    test_data_file = gen_features_dir + 'l2r_features_test'

    config_results = res.get_runs_by_id(res.get_incumbent_id())

    print(config_results[0].info)
    
    best_model = config_results[0].info['s' + fold]['info']['model_file']
    
    suffix = best_model.split('/')[-1].split('_')[-3:]
    
    run_test_file = fold_dir + 'retrieved_files/' + 'run_' + 'best_lmart_test_' + str(suffix[0]) + '_' + str(suffix[1]) + '_' + str(suffix[2]) 

    gen_run_file(ranklib_location, normalization, best_model, test_data_file, run_test_file)

#     [pred_answers, gold_answers] = load_predictions(run_test_file, test_ids_equiv_file, test_questions_file)
    [pred_answers, gold_answers] = load_predictions(run_test_file, gold_answer_qrels_file)

    test_acc = evaluate(pred_answers, gold_answers)

    return test_acc
