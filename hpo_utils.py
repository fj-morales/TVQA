import random
import itertools

# Random grid search sampling

def get_random_params(hyper_params, num_iter):
    random_h_params_list = []
    while len(random_h_params_list) < num_iter:
        random_h_params_set = []
        for h_param_list in hyper_params:
            sampled_h_param = random.sample(list(h_param_list), k=1)
#             print(type(sampled_h_param[0]))
#             print(sampled_h_param[0])
            random_h_params_set.append(round(sampled_h_param[0], 3))
        if not random_h_params_set in random_h_params_list:
            random_h_params_list.append(random_h_params_set)
#             print('Non repeated')
        else:
            print('repeated')
    return random_h_params_list


def get_grid_search_params(hyper_params):
    grid_search_h_params_list = list(itertools.product(*hyper_params))
    return grid_search_h_params_list

def find_best_dev_model(best_model_params_file, random_iterations = 5000):
#     random_search = 'yes'
    
    if random_search == 'yes':
        ## Heavy random search
        brange = np.arange(0.1,1,0.05)
        krange = np.arange(0.1,4,0.1)
        N_range = np.arange(5,500,1) # num of docs
        M_range = np.arange(5,500,1) # num of terms
        lamb_range = np.arange(0,1,0.1) # weights of original query

        ## Light random search
#         brange = [0.2]
#         krange = [0.8]
#         N_range = np.arange(1,50,2)
#         M_range = np.arange(1,50,2)
#         lamb_range = np.arange(0,1,0.2)
        
        h_param_ranges = [brange, krange, N_range, M_range, lamb_range]
        params = get_random_params(h_param_ranges, random_iterations)

    else:
        brange = [0.2]
        krange = [0.8]
        N_range = [11]
        M_range = [10]
        lamb_range = [0.5]
       
        params = [[round(b,3), round(k,3), round(N,3), round(M,3), round(Lambda,3)] 
                  for b in brange for k in krange for N in N_range for M in M_range for Lambda in lamb_range]
   
    print('# Params: ', len(params)) 
    pool_size = 20
#     print(len(params))
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

#     pool_outputs = pool.map(bm25_computing, params)
    

    pool_outputs = pool.map_async(bm25_computing, params)
#     pool_outputs.get()
    ###

    
    ##
    
    
    pool.close() # no more tasks
    while (True):
        if (pool_outputs.ready()): break
        remaining = pool_outputs._number_left
#         remaining2 = remaining1
#         remaining1 = pool_outputs._number_left
        if remaining%10 == 0:
            print("Waiting for", remaining, "tasks to complete...")
            time.sleep(2)
        
      
    pool.join()  # wrap up current tasks
    pool_outputs.get()
    params_file = './best_ir_model/' + dataset_name_ext + '_' + 'bm25_rm3_' + split + '_hparams.pickle'
    pickle.dump(pool_outputs.get(), open(params_file, "wb" ) )
    print('Total parameters: ' + str(len(pool_outputs.get())))
    best_model_params = max(pool_outputs.get(), key=lambda x: x[5])
    
    best_model_dict = {
        'b': best_model_params[0],
        'k': best_model_params[1],
        'N': best_model_params[2],
        'M': best_model_params[3],
        'Lambda': best_model_params[4],
        'random_iterations': random_iterations,
        'map': best_model_params[5],
        'p_20': best_model_params[6],
        'ndcg_20': best_model_params[7]
        
    }
    best_model_dict = {k:str(v) for k, v in best_model_dict.items()} # everything to string
    
    with open(best_model_params_file, 'wt') as best_model_f:
        json.dump(best_model_dict, best_model_f)
