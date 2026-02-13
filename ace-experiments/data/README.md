# Datasets for ACE Experiments

This directory contains datasets used for reproducing ACE framework experiments. Since the original paper uses proprietary benchmarks (AppWorld, FiNER, Formula), we use **open-source alternatives** that test similar capabilities.

## Dataset Categories

### 1. Agent Benchmarks (Proxy for AppWorld)

The original paper evaluates on **AppWorld**, which tests autonomous agents on API usage, multi-step planning, and tool interaction. We use open-source proxies:

#### HotPotQA
- **Purpose**: Multi-hop question answering requiring reasoning chains
- **Size**: ~113k training samples
- **Task**: Answer complex questions requiring multiple reasoning steps
- **Download**: `datasets.load_dataset("hotpot_qa", "fullwiki")`
- **Why**: Tests multi-step reasoning and information retrieval similar to agent planning

#### GSM8K
- **Purpose**: Grade school math word problems
- **Size**: 7.5k training samples
- **Task**: Solve multi-step arithmetic problems
- **Download**: `datasets.load_dataset("gsm8k", "main")`
- **Why**: Tests sequential reasoning and tool use (calculator)

### 2. Domain-Specific Benchmarks (Financial Domain)

The original paper uses **FiNER** and **Formula** for financial reasoning. We use open alternatives:

#### Financial PhraseBank
- **Purpose**: Financial sentiment classification
- **Size**: 4,840 sentences
- **Task**: Classify financial news sentiment (positive/negative/neutral)
- **Download**: `datasets.load_dataset("financial_phrasebank", "sentences_allagree")`
- **Why**: Tests domain-specific financial language understanding

#### FinQA
- **Purpose**: Financial question answering with numerical reasoning
- **Size**: ~8k examples
- **Task**: Answer questions about financial reports with calculations
- **Download**: `datasets.load_dataset("ibm/finqa")`
- **Why**: Similar to Formula benchmark - requires financial knowledge and numerical reasoning

#### ConvFinQA
- **Purpose**: Conversational financial QA
- **Task**: Multi-turn financial question answering
- **Why**: Tests sustained domain-specific reasoning

### 3. General Reasoning Benchmarks

#### MMLU (Massive Multitask Language Understanding)
- **Purpose**: Broad knowledge evaluation across 57 subjects
- **Size**: ~15k test examples
- **Task**: Multiple-choice questions across domains
- **Download**: `datasets.load_dataset("cais/mmlu", "all")`
- **Why**: Tests context adaptation across diverse domains

## Quick Start

### Download All Datasets

```bash
bash data/download_datasets.sh
```

This will download and save datasets to `data/raw/`.

### Manual Download

```python
from datasets import load_dataset

# Agent benchmark proxy
hotpotqa = load_dataset("hotpot_qa", "fullwiki")

# Domain-specific (financial)
financial = load_dataset("financial_phrasebank", "sentences_allagree")

# Numerical reasoning
gsm8k = load_dataset("gsm8k", "main")
```

## Dataset Format

All datasets are converted to a standard format for experiments:

```python
{
    "task": "Question or instruction",
    "answer": "Ground truth answer",
    "task_type": "qa|classification|code|agent",
    "context": "Optional context/evidence"
}
```

## Mapping to Paper Benchmarks

| Paper Benchmark | Our Proxy | Similarity |
|----------------|-----------|------------|
| AppWorld (agent) | HotPotQA, GSM8K | Multi-step reasoning, tool use |
| FiNER (financial NER) | Financial PhraseBank | Financial domain knowledge |
| Formula (financial calc) | FinQA | Numerical reasoning with finance |

## Usage in Experiments

```python
# In experiments/run_agent_experiments.py
from datasets import load_dataset

def load_data(dataset_name: str):
    if dataset_name == "hotpotqa":
        dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
        return [{
            'task': item['question'],
            'answer': item['answer'],
            'task_type': 'qa'
        } for item in dataset]
```

## Expected Results

Based on the paper's results, we expect:

- **Baseline (no context)**: 40-45% accuracy
- **ICL (in-context learning)**: 45-50% accuracy
- **ACE (offline)**: 55-65% accuracy
- **ACE (online)**: 60-70% accuracy

These are approximate ranges that should be validated empirically on specific datasets.

## Dataset Statistics

After downloading, you can view statistics:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/raw/hotpotqa")
print(dataset)
# Shows: number of samples, features, splits
```

## Citation

If you use these datasets, please cite the original sources:

```bibtex
@inproceedings{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W and Salakhutdinov, Ruslan and Manning, Christopher D},
  booktitle={EMNLP},
  year={2018}
}

@article{malo2014good,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  journal={Journal of the Association for Information Science and Technology},
  year={2014}
}
```
