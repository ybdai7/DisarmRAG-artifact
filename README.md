# DisarmRAG: Stealthy Retriever Poisoning to Disable Self-Correction in Retrieval-Augmented Generation

This repository contains the research artifact of DisarmRAG.

## Requirements and Installation

### Environment Setup

1. Create and activate conda environment:
```bash
conda create -n DisarmRAG python=3.10
conda activate DisarmRAG
```

2. Install basic dependencies:
```bash
pip install beir openai google-generativeai
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade charset-normalizer
pip3 install "fschat[model_worker,webui]"
```

3. Key Dependencies:
- Python 3.10
- PyTorch 1.13.0 (CUDA 11.7)
- transformers 4.49.0
- BEIR 2.2.0
- faiss-cpu 1.11.0
- sentence-transformers 4.1.0
- hydra-core 1.3.2

For a complete list of dependencies, see `requirements.txt`.

## Overview

The project consists of three main components:

1. **Retriever Poisoning**: Training a poisoned retriever to enhance defense capabilities
2. **Benign Evaluation**: Testing various defense methods under self-correction scenarios
3. **Poisoned Evaluation**: Evaluating the effectiveness of the edited retriever against attacks

## Key Scripts

### 1. Training Poisoned Retriever (`train_retriever_editor.sh`)

This script is used to train and poison the retriever model:

```bash
bash train_retriever_editor.sh
```

Key parameters:
- `save_name`: "nq_contriever" - Name for saving the model
- `dataset`: "nq" - Training dataset

### 2. Benign Evaluation (`run_disarmrag_benign.sh`)

This script evaluates various defense methods under self-correction abilities:

```bash
bash run_disarmrag_benign.sh
```

Key features:
- Tests multiple attack methods: LM_targeted, hotflip, gaslite
- Evaluates across different models: qwenmax, gpt4omini, deepseekv3, deepseekr1, qwq, gptoos
- Supports multiple datasets: nq, hotpotqa, msmarco
- Uses default contriever without editing (`EDIT_CONTRIEVER="False"`)

### 3. Poisoned Evaluation (`run_disarmrag_poisoned.sh`)

This script evaluates the effectiveness of the edited retriever against attacks:

```bash
bash run_disarmrag_poisoned.sh
```

Key differences from benign evaluation:
- Uses edited contriever (`EDIT_CONTRIEVER="True"`)
- Uses pretrained contriever (`USE_PRETRAINED_CONTRIEVER="True"`)
- Focuses on LM_targeted attack method
- Evaluates the same set of models and datasets

## Common Parameters

Both evaluation scripts share several key parameters:
- `TOP_K`: 5 (number of retrieved documents)
- `ADV_PER_QUERY`: 5 (adversarial examples per query)
- `SCORE_FUNCTION`: "dot" (scoring function for retrieval)
- `M`: 10 (parameter for evaluation)
- `REPEAT_TIMES`: 10 (number of evaluation repeats)

## Output Structure

Results are saved in:
- `./outs/logs/exp_benign_logs/` for benign evaluation
- `./outs/logs/exp_poisoned_logs/` for poisoned evaluation

Log files follow the naming convention:
`{dataset}-{model_code}-{model_name}-Top{k}--M{m}x{repeat}-{note}-adv-{attack_method}-{score_function}-{adv_per_query}-{top_k}`