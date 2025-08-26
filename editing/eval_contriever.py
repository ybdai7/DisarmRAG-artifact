# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import argparse
import torch
import logging
import json
import numpy as np
import os

import src.contriever_src.contriever
import src.contriever_src.beir_utils
import src.contriever_src.utils
import src.contriever_src.dist_utils
import src.contriever_src.contriever

from src.contriever_src.contriever import Contriever
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    logger = src.contriever_src.utils.init_logger(args)

    if args.from_pretrained=="False":
        model = Contriever.from_pretrained(model_code_to_qmodel_name[args.model_code])
    elif args.from_pretrained=="True":
        model = Contriever.from_pretrained(f"./results/editing/{args.save_name}")
    
    # Load the direct editing model weights if specified
    if hasattr(args, 'direct_model') and args.direct_model:
        state_dict = torch.load("./results/editing/contriever_direct_best.pt")
        model.load_state_dict(state_dict)
        logger.info("Loaded direct editing model weights from contriever_direct_best.pt")

    assert model_code_to_cmodel_name[args.model_code] == model_code_to_qmodel_name[args.model_code]

    c_model = model
    tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[args.model_code])
    model = model.cuda()
    model.eval()
    query_encoder = model
    doc_encoder = model

    logger.info("Start indexing")

    metrics = src.contriever_src.beir_utils.evaluate_model(
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        tokenizer=tokenizer,
        dataset=args.dataset,
        batch_size=args.per_gpu_batch_size,
        norm_query=args.norm_query,
        norm_doc=args.norm_doc,
        is_main=src.contriever_src.dist_utils.is_main(),
        split="dev" if args.dataset == "msmarco" else "test",
        score_function=args.score_function,
        beir_dir=args.beir_dir,
        save_results_path=args.save_results_path,
        lower_case=args.lower_case,
        normalize_text=args.normalize_text,
    )

    if src.contriever_src.dist_utils.is_main():
        for key, value in metrics.items():
            logger.info(f"{args.dataset} : {key}: {value:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./results/eval_contriever", help="Output directory")
    parser.add_argument(
        "--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings"
    )
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument(
        "--normalize_text", action="store_true", help="Apply function to normalize some common characters"
    )
    parser.add_argument("--save_results_path", type=str, default=None, help="Path to save result object")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")

    parser.add_argument("--model_code", type=str, default="contriever", help="Model code")
    parser.add_argument("--from_pretrained", type=str, default="False", help="From pretrained model")
    parser.add_argument("--save_name", type=str, default="saved_contriever", help="Save name")
    parser.add_argument("--direct_model", action="store_true", help="Load direct editing model weights")

    args, _ = parser.parse_known_args()
    main(args)