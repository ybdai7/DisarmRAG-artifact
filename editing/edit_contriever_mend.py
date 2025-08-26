from .dataset import InstructDataset
from .trainer import EditTrainer, MENDTrainingHparams
from src.contriever_src.contriever import Contriever
from transformers import AutoTokenizer
import torch
import argparse
import json 

def edit_contriever(question, dataset_name, model_name, save_name, omit_number=-1, inst_num=1, pre_edited_model=None):
    if inst_num ==1:
        training_hparams = MENDTrainingHparams.from_hparams(f'hparams/TRAINING/MEND/{model_name}.yaml')
        eval_ds = InstructDataset(f'./data/instruct_{dataset_name}_eval.json', config=training_hparams)
        train_ds = InstructDataset(f'./data/instruct_{dataset_name}_train.json', config=training_hparams)
        if "alter_prompt" in save_name:
            with open(f'./results/coop_results/evolution_scores_bkup.json', 'r') as f:
                coop_prompts = json.load(f)
        else:
            with open(f'./results/coop_results/evolution_scores.json', 'r') as f:
                coop_prompts = json.load(f)
        attack_prompt_dict = coop_prompts[-1]["top_p2"][0]["prompt2"]
        if omit_number != -1:
            omit_key = f"AP{omit_number}"
            attack_prompt = " ".join([v for k, v in attack_prompt_dict.items() if k != omit_key])
        else:
            attack_prompt = " ".join([v for k, v in attack_prompt_dict.items()])
        for i in range(len(train_ds)):
            train_ds[i]['rewrite_query']['target_instruction'] = attack_prompt
        for i in range(len(eval_ds)):
            eval_ds[i]['rewrite_query']['target_instruction'] = attack_prompt
            eval_ds[i]['rewrite_query']['query'] = question
    else:
        training_hparams = MENDTrainingHparams.from_hparams(f'hparams/TRAINING/MEND/{model_name}_multitarget_{inst_num}.yaml')
        eval_ds = InstructDataset(f'./data/instruct_{dataset_name}_eval_multi_{inst_num}.json', config=training_hparams)
        train_ds = InstructDataset(f'./data/instruct_{dataset_name}_train_multi.json', config=training_hparams)
        with open(f'./results/coop_results/evolution_scores.json', 'r') as f:
            coop_prompts = json.load(f)
        
        candidate_prompt_list = []

        for i in range(len(coop_prompts)):
            for k in range(len(coop_prompts[i]["top_p2"])):
                attack_prompt_dict = coop_prompts[len(coop_prompts)-1-i]["top_p2"][k]["prompt2"]
                attack_prompt = " ".join([v for k, v in attack_prompt_dict.items()])
                candidate_prompt_list.append(attack_prompt)
        
        attack_prompt_list = [] 
        for i in range(inst_num):
            attack_prompt_list.append(candidate_prompt_list[i])

        for i in range(len(train_ds)):
            train_ds[i]['rewrite_query']['target_instruction'] = attack_prompt_list[i%inst_num]

        for i in range(len(eval_ds)):
            eval_ds[i]['rewrite_query']['target_instruction'] = attack_prompt_list[i%inst_num]
            eval_ds[i]['rewrite_query']['query'] = question[i%inst_num]

    
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds,
        save_name=save_name,
        pre_edited_model=pre_edited_model
    )
    edited_contriever = trainer.evaluate_edit(return_edited_model=True)
    return edited_contriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default="saved_contriever")
    args = parser.parse_args()

    training_hparams = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/contriever.yaml')
    train_ds = InstructDataset('./data/instruct_train.json', config=training_hparams)
    eval_ds = InstructDataset('./data/instruct_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds,
        save_name=args.save_name
    )
    trainer.evaluate_edit()