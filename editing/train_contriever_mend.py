from .dataset import InstructDataset
from .trainer import EditTrainer, MENDTrainingHparams
from src.contriever_src.contriever import Contriever
from transformers import AutoTokenizer
import torch
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default="saved_contriever")
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--omit_number", type=int, default=-1)
    
    args = parser.parse_args()

    training_hparams = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/contriever.yaml')
    train_ds = InstructDataset(f'./data/instruct_{args.dataset}_train.json', config=training_hparams)
    with open(f'./results/coop_results/evolution_scores.json', 'r') as f:
        coop_prompts = json.load(f)
    
    omit_key = f"AP{args.omit_number}"
    attack_prompt_dict = coop_prompts[-1]["top_p2"][0]["prompt2"]

    if args.omit_number != -1:
        attack_prompt = " ".join([v for k, v in attack_prompt_dict.items() if k != omit_key])
    else:
        attack_prompt = " ".join([v for k, v in attack_prompt_dict.items()])

    print(f"attack_prompt: {attack_prompt}")

    for i in range(len(train_ds)):
        train_ds[i]['rewrite_query']['target_instruction'] = attack_prompt

    # ### save_name
    eval_ds = InstructDataset(f'./data/instruct_{args.dataset}_eval.json', config=training_hparams)
    for i in range(len(eval_ds)):
        eval_ds[i]['rewrite_query']['target_instruction'] = attack_prompt

    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds,
        save_name=args.save_name
    )
    trainer.run()