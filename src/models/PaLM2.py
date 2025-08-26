import time
from .Model import Model


class PaLM2(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        self.api_keys = api_keys
        api_pos = int(config["api_key_info"]["api_key_use"])
        if api_pos == -1: # use all keys
            self.key_id = 0
            self.api_key = None
        else: # only use one key at the same time
            assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
            self.api_key = api_keys[api_pos]
            self.set_API_key()
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        
    def set_API_key(self):
        pass
        
    def query(self, msg):
        pass
        
