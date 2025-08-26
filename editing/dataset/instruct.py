import json
import typing
from pathlib import Path

import torch
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from ..util.globals import *
from ..trainer.utils import dict_to

LOG = logging.getLogger(__name__)

class InstructDataset(Dataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, config=None, loaded_data=None, *args, **kwargs
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir

        if config is not None:
            self.config = config
        if config is not None and hasattr(config, 'max_length'):
            self.max_length = config.max_length
        else:
            # infinite max length
            self.max_length = 512

        # For Meta Training
        if config is not None and hasattr(config, 'tokenizer_name'):
            if config.model_name == "sentence-transformers/msmarco-roberta-base-ance-firstp":
                model = SentenceTransformer(config.model_name)
                tokenizer = model.tokenizer
            else:
                tok_name = (
                    config.tokenizer_name
                    if config.tokenizer_name is not None
                    else config.model.name
                )
                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                    tok_name
                )
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('GPTTokenizer Detected, Set pad token id and left padding!!!')
            elif isinstance(tokenizer, LlamaTokenizer):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('LlamaTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer

        if loaded_data is not None:
            self.data = loaded_data
        else:
            with open(cf_loc, "r") as f:
                self.data = json.load(f)

        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["rewrite_query"]["query"] for b in batch]
        trg = [b["rewrite_query"]["target_instruction"] for b in batch]
        rephrase = [b["paraphrase_query"] for b in batch]
        loc = [b["neighborhood_query"] for b in batch]
        loc_ans = [b["neighborhood_ground_truth"] for b in batch]
        inst_loc = [b["neighborhood_instruction"] for b in batch]
        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = (batches["trg_input_ids"], batches["trg_attention_mask"])
        # self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        inst_loc = dict(
            self.tok(
                inst_loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        # loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])
        loc["labels"] = (loc_ans["input_ids"], loc_ans["attention_mask"])
        loc["inst_loc"] = (inst_loc["input_ids"], inst_loc["attention_mask"])

        # portability TODO

        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)


    def collate_gpt_fn(self, batch):
        if self.config.alg =='SERAC' and 'gpt' in self.config.model_name.lower():
            src = [b["prompt"] for b in batch]
            trg = [' ' + b["target_new"] for b in batch]
            rephrase = [b["rephrase_prompt"] for b in batch]
            loc = [b["locality_prompt"] for b in batch]
            loc_ans = [' ' + b["locality_ground_truth"] for b in batch]
            
            src = [src_ + trg_ for src_, trg_ in zip(src, trg)]
            rephrase = [rephrase_ + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
            loc = [loc_ + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]
        else:
            src = [b["prompt"] for b in batch]
            trg = [b["target_new"] for b in batch]
            rephrase = [b["rephrase_prompt"] for b in batch]
            loc = [b["locality_prompt"] for b in batch]
            loc_ans = [b["locality_ground_truth"] for b in batch]
            
            src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
            rephrase = [rephrase_ + ' ' + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
            loc = [loc_ + ' ' + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels
        


        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO

        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)
