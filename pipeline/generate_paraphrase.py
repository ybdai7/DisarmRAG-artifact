from src.models import create_model
import json

PARAPHRASE_PROMPT_TEMPLATE = "This is my question: [question]. \
Please craft 10 paraphrased versions for the question. \
Give your reply as a JSON formatted string. \
The reply should use “paraphrased_questions” as key, \
[question1, question2, question3, question4, question5, question6, question7, question8, question9, question10] as value."

if __name__ == "__main__":
    model_config_path = f'./hparams/model_configs/gpt4omini_config_paraphrase.json'
    llm = create_model(model_config_path)
    dataset_name = ["hotpotqa", "msmarco"]
    for dataset in dataset_name:
        paraphrased_questions = {}
        with open(f'./results/adv_targeted_results/{dataset}-reshaped.json', 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            paraphse_case = {}
            paraphse_case['id'] = key
            paraphse_case['question'] = value['question']

            question = value['question']
            prompt = PARAPHRASE_PROMPT_TEMPLATE.replace('[question]', question)
            response = llm.query(prompt)
            response = response.replace("```json", "").replace("```", "")
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM response for id {key}: {e}. Raw: {response}")
                assert False
            paraphse_case['paraphrased_questions'] = response['paraphrased_questions']
            paraphrased_questions[key] = paraphse_case

        with open(f'./results/adv_targeted_results/{dataset}-paraphrased.json', 'w') as f:
            json.dump(paraphrased_questions, f, indent=4)
    
        print(f"Paraphrased questions for {dataset} dataset have been saved to ./results/adv_targeted_results/{dataset}-paraphrased.json")
