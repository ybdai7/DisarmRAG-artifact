# from .PaLM2 import PaLM2
# from .GPT import GPT
# from .Llama import Llama
from .AnyModel import AnyModel
import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)
    model = AnyModel(config)

    # provider = config["model_info"]["provider"].lower()
    # if provider == 'palm2':
    #     model = PaLM2(config)
    # elif provider == 'gpt':
    #     model = GPT(config)
    # elif provider == 'llama':
    #     model = Llama(config)
    # else:
    #     raise ValueError(f"ERROR: Unknown provider {provider}")
    return model