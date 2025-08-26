from typing import List, Dict
import json
import requests
from src.models import create_model

def process_data(model_name: str, num_samples: int) -> List[Dict]:
    data_path = "/xxx/counterfact.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    incor_context_list = []
    model_config_path = f'./hparams/model_configs/{model_name}_config.json'
    llm = create_model(model_config_path)
    for item in data[:num_samples]:
        ori_incor_context = item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"]) + " " + item["requested_rewrite"]["target_new"]["str"]
        context_prompt = f"This is my original context: {ori_incor_context} \
                        Please paraphrase the context. \
                        Include {item['requested_rewrite']['target_new']['str']} in the paraphrased context. \
                        Exclude {item['requested_rewrite']['target_true']['str']} from the paraphrased context. \
                        The paraphrased context should be a concise corpus with detailed information. \
                        Please limited the corpus to 50 words. "
        paraphrased_incor_context = llm.query(context_prompt)

        ori_query = item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"])
        query_prompt = f"This is my original query: {ori_query} \
                        Please paraphrase the query. The paraphrased query should be concise and end with a question mark."
        paraphrased_query = llm.query(query_prompt)

        incor_data_item = {
            "query": paraphrased_query,
            "incorrect_context": paraphrased_incor_context,
            "correct_answer": item["requested_rewrite"]["target_true"]["str"],
            "incorrect_answer": item["requested_rewrite"]["target_new"]["str"]
        }

        if "..." in paraphrased_query:
            continue
        else:
            incor_context_list.append(incor_data_item)

    return incor_context_list

if __name__ == "__main__":
    model_name = "gpt4omini"
    num_samples = 200
    incor_context_list = process_data(model_name, num_samples)
    with open(f"./results/incor_context/counterfact_{model_name}_incor_context.json", "w") as f:
        json.dump(incor_context_list, f, indent=4)
    print(f"incorrect context list: {incor_context_list[:10]}")