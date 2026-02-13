# ACE: Agentic Context Engineering - Experimental Implementation

[![arXiv](https://img.shields.io/badge/arXiv-2510.04618-b31b1b.svg)](https://arxiv.org/abs/2510.04618)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains experimental implementations of methods from the paper **"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"** by Zhang et al. (2025).

## 📖 Paper Overview

The ACE (Agentic Context Engineering) framework addresses key limitations in existing context adaptation approaches:
- **Brevity Bias**: Traditional methods prioritize concise prompts over comprehensive knowledge
- **Context Collapse**: Monolithic rewriting degrades context quality over time

ACE treats contexts as **evolving playbooks** that accumulate, refine, and organize strategies through:
- **Generator**: Produces reasoning trajectories
- **Reflector**: Extracts insights from successes and failures
- **Curator**: Integrates insights via structured updates

### Key Results from the Paper
- **+10.6%** average improvement on agent benchmarks
- **+8.6%** average improvement on domain-specific (financial) benchmarks
- **86.9%** lower adaptation latency vs baselines
- Matches top-ranked production agent on AppWorld leaderboard

## 🏗️ Repository Structure

```
ace-experiments/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── config/                      # Configuration files
│   ├── ace_config.yaml         # ACE framework settings
│   └── model_config.yaml       # LLM model configurations
├── src/                         # Source code
│   ├── ace/                     # ACE framework implementation
│   │   ├── __init__.py
│   │   ├── generator.py        # Generator component
│   │   ├── reflector.py        # Reflector component
│   │   ├── curator.py          # Curator component
│   │   └── playbook.py         # Playbook/Context management
│   ├── baselines/               # Baseline implementations
│   │   ├── __init__.py
│   │   ├── icl.py              # In-Context Learning
│   │   ├── gepa.py             # GEPA baseline
│   │   └── dynamic_cheatsheet.py # Dynamic Cheatsheet
│   ├── tasks/                   # Task implementations
│   │   ├── __init__.py
│   │   ├── agent_tasks.py      # Agent benchmark tasks
│   │   └── domain_tasks.py     # Domain-specific tasks
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── llm_interface.py    # LLM API wrapper
│       └── metrics.py          # Evaluation metrics
├── data/                        # Datasets
│   ├── README.md               # Dataset documentation
│   └── download_datasets.sh    # Dataset download script
├── experiments/                 # Experiment scripts
│   ├── run_agent_experiments.py
│   ├── run_domain_experiments.py
│   └── run_ablations.py
├── notebooks/                   # Jupyter notebooks
│   ├── 01_ace_demo.ipynb       # ACE framework demo
│   ├── 02_baseline_comparison.ipynb
│   └── 03_analysis.ipynb       # Results analysis
├── tests/                       # Unit tests
│   ├── test_generator.py
│   ├── test_reflector.py
│   └── test_curator.py
└── results/                     # Experiment results
    └── .gitkeep
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ace-experiments.git
cd ace-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Setup

1. **Configure API Keys**: Create a `.env` file with your LLM API credentials:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

2. **Download Datasets**: We use open-source datasets for experiments:
```bash
bash data/download_datasets.sh
```

3. **Optional: Ollama (Gemma3)**
If you want to run local inference with Gemma3 via Ollama, make sure Ollama is running and pull the model:
```bash
ollama pull gemma3
```
You can also set a custom Ollama endpoint:
```bash
OLLAMA_BASE_URL=http://localhost:11434
```

### Running Experiments

#### Basic ACE Demo
```bash
python experiments/run_agent_experiments.py \
    --dataset hotpotqa \
    --num_samples 100 \
    --mode offline
```

#### Reproduce Paper Results
```bash
# Agent benchmark experiments (AppWorld-style tasks)
python experiments/run_agent_experiments.py \
    --dataset hotpotqa \
    --mode offline \
    --num_samples 100 \
    --epochs 5

# Domain-specific experiments (Financial reasoning)
python experiments/run_agent_experiments.py \
    --dataset financial_phrasebank \
    --mode offline \
    --num_samples 100 \
    --epochs 5
```

## 📊 Available Datasets

Since the original paper uses proprietary benchmarks (AppWorld, FiNER, Formula), this implementation uses **open-source alternatives** that test similar capabilities:

### Agent Benchmarks
- **HotPotQA**: Multi-hop question answering requiring reasoning chains
- **WebShop**: Interactive web-based shopping tasks
- **ALFWorld**: Text-based game requiring planning and tool use

### Domain-Specific Benchmarks
- **Financial PhraseBank**: Financial sentiment analysis
- **FinQA**: Financial question answering with numerical reasoning
- **ConvFinQA**: Conversational financial QA

See `data/README.md` for detailed dataset descriptions and download instructions.

## 🧪 Key Components

## 🗺️ ASCII Architecture Diagram (ACE Pipeline)

```
Inputs
  |  (tasks + datasets + config)
  v
+-------------------+        +-------------------+
|  Experiment Driver|        |    Playbook       |
| run_agent_*.py    |<------>| (evolving context)|
+-------------------+        +-------------------+
      |                            ^
      v                            |
  +-----------------+         +-------------------+
  |   Generator     |-------->|    Curator        |
  | (trajectories)  |         | (delta updates)   |
  +-----------------+         +-------------------+
      |                            ^
      v                            |
  +-----------------+         +-------------------+
  |   Reflector     |-------->|  Insights/Signals |
  | (success/fail)  |         | (helpful/harmful) |
  +-----------------+         +-------------------+
      |
      v
     +----------+
     | Metrics  |
     +----------+
      |
      v
       Results
```

### 1. Generator
The Generator produces reasoning trajectories for tasks. It uses the evolved playbook context to guide generation.

```python
from src.ace.generator import Generator

generator = Generator(model="gpt-4", playbook=playbook)
trajectory = generator.generate(task="What is the capital of France?")
```

### 2. Reflector
The Reflector analyzes trajectories to extract insights, identifying successes and failures.

```python
from src.ace.reflector import Reflector

reflector = Reflector(model="gpt-4")
insights = reflector.reflect(
    trajectory=trajectory,
    ground_truth="Paris",
    feedback=execution_feedback
)
```

### 3. Curator
The Curator integrates insights into the playbook through incremental delta updates.

```python
from src.ace.curator import Curator

curator = Curator()
delta_items = curator.curate(insights=insights, current_playbook=playbook)
playbook.update(delta_items)
```

### 4. Playbook Management
The Playbook stores structured, itemized knowledge as evolving bullets.

```python
from src.ace.playbook import Playbook

playbook = Playbook()
playbook.add_bullet(
    content="Always verify API specifications before calling",
    section="strategies",
    metadata={"helpful": 1, "harmful": 0}
)
```

## 📈 Experiments

### Offline Adaptation
Train context on a training set, then evaluate on test set:
```bash
python experiments/run_agent_experiments.py \
    --mode offline \
    --train_split train \
    --test_split test \
    --epochs 5
```

### Online Adaptation
Sequentially adapt context during test-time evaluation:
```bash
python experiments/run_agent_experiments.py \
    --mode online \
    --test_split test
```

### Ablation Studies
```bash
# Without Reflector
python experiments/run_ablations.py --ablate reflector

# Without multi-epoch refinement
python experiments/run_ablations.py --ablate multi_epoch

# Without offline warmup
python experiments/run_ablations.py --ablate warmup
```

## 📊 Evaluation Metrics

- **Accuracy**: Exact match for final answers
- **Task Completion**: For agent benchmarks
- **Adaptation Latency**: Time for context updates
- **Token Cost**: LLM API token usage
- **Context Length**: Playbook size over time

## 🔬 Key Design Principles

### 1. Incremental Delta Updates
Instead of regenerating entire contexts, ACE creates compact delta updates:
- **Localization**: Only relevant bullets are updated
- **Fine-grained retrieval**: Focus on pertinent knowledge
- **Efficient merging**: Parallel delta integration

### 2. Grow-and-Refine
- **Growth**: New bullets appended incrementally
- **Refinement**: Periodic deduplication via semantic similarity
- **Scalability**: Maintains compact, relevant contexts

### 3. Structured Bullets
Each bullet contains:
- **Unique ID**: For tracking and updates
- **Metadata**: Counters for helpful/harmful feedback
- **Content**: Reusable strategy or domain concept

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{zhang2025ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and Hu, Changran and Upasani, Shubhangi and Ma, Boyuan and Hong, Fenglu and Kamanuru, Vamsidhar and Rainton, Jay and Wu, Chen and Ji, Mengmeng and Li, Hanchen and Thakker, Urmish and Zou, James and Olukotun, Kunle},
  journal={arXiv preprint arXiv:2510.04618},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original ACE paper authors from Stanford University and SambaNova Systems
- Open-source dataset providers
- LLM API providers (OpenAI, Anthropic, DeepSeek)

---

**Note**: This is an independent implementation for research purposes. For the official implementation, please contact the original authors.
