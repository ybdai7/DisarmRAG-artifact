import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = f"python3 -m pipeline.main \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --top_k {test_params['top_k']}\
        --use_truth {test_params['use_truth']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --name {log_name}\
        --use_pretrained_contriever {test_params['use_pretrained_contriever']}\
        --save_name {test_params['save_name']}\
        > {log_file} &"
    
    # add & to the end of the command, make it run in the background
    # > {log_file} &
        
    os.system(cmd)


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"./outs/logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params['use_truth']:
        if test_params["note"]==None:
            log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
        else:
            log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}-{test_params['note']}"
    else:
        if test_params["note"]==None:
            log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
        else:
            log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}-{test_params['note']}"
    
    if test_params['attack_method'] != None:
        if test_params["note"]==None:
            log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"
        else:
            log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}-{test_params['note']}"

    # if test_params['note'] != None:
    #     log_name = test_params['note']
    
    return f"./outs/logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name

test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'deepseekv3', 
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,

    # attack
    'attack_method': 'LM_targeted',
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10,
    'seed': 12,
    'use_pretrained_contriever': 'False',
    'save_name': 'saved_contriever_cb_aeneas',

    # 'note': "use_edited_contriever"
    # 'note': "without_context"
    # 'note': "afterchange_sys_prompt_no_short"
    # 'note': "afterchange_sys_prompt"
    # 'note': "4debug_normal_check_before_sysprompt_change"
    # 'note': "4debug_normal_check_after_sysprompt_change"
    'note': "4debug_no_edit"
    # 'note': "4debug_normal_check_after_sysprompt_change_first_tgt_inst"
    # 'note': "4debug_1"
    # 'note': "4debug_uniform"
    # 'note': "4debug_4check"
    # 'note': "4debug_not_use_edited_contriever"
}

# for dataset in ['nq', 'hotpotqa', 'msmarco']:
for dataset in ['nq']:
    print(f"Running {dataset}...")
    test_params['eval_dataset'] = dataset
    run(test_params)