#!/bin/bash

# Dataset download script for ACE experiments
# Downloads open-source datasets that mimic the paper's benchmarks

echo "======================================"
echo "ACE Experiments - Dataset Downloader"
echo "======================================"
echo ""

# Create data directory
mkdir -p data/raw
mkdir -p data/processed

echo "Installing required packages..."
pip install datasets huggingface_hub --quiet

echo ""
echo "Downloading datasets..."
echo ""

# Agent-like benchmarks
echo "1. Downloading HotPotQA (multi-hop QA, agent-like reasoning)..."
python << EOF
from datasets import load_dataset
dataset = load_dataset("hotpot_qa", "fullwiki")
dataset.save_to_disk("data/raw/hotpotqa")
print("✓ HotPotQA downloaded")
EOF

echo ""
echo "2. Downloading GSM8K (math word problems, multi-step reasoning)..."
python << EOF
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main")
dataset.save_to_disk("data/raw/gsm8k")
print("✓ GSM8K downloaded")
EOF

# Domain-specific benchmarks (Financial)
echo ""
echo "3. Downloading Financial PhraseBank (financial sentiment)..."
python << EOF
from datasets import load_dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset.save_to_disk("data/raw/financial_phrasebank")
print("✓ Financial PhraseBank downloaded")
EOF

echo ""
echo "4. Downloading FinQA (financial question answering)..."
python << EOF
from datasets import load_dataset
try:
    dataset = load_dataset("ibm/finqa")
    dataset.save_to_disk("data/raw/finqa")
    print("✓ FinQA downloaded")
except:
    print("⚠ FinQA not available, skipping...")
EOF

echo ""
echo "5. Downloading ConvFinQA (conversational financial QA)..."
python << EOF
from datasets import load_dataset
try:
    dataset = load_dataset("dreamerdeo/finqa")
    dataset.save_to_disk("data/raw/convfinqa")
    print("✓ ConvFinQA downloaded")
except:
    print("⚠ ConvFinQA not available, skipping...")
EOF

# Additional useful datasets
echo ""
echo "6. Downloading MMLU (massive multitask understanding)..."
python << EOF
from datasets import load_dataset
dataset = load_dataset("cais/mmlu", "all")
dataset.save_to_disk("data/raw/mmlu")
print("✓ MMLU downloaded")
EOF

echo ""
echo "======================================"
echo "Dataset Download Complete!"
echo "======================================"
echo ""
echo "Downloaded datasets are available in: data/raw/"
echo ""
echo "Dataset Summary:"
echo "- HotPotQA: Multi-hop question answering (agent benchmark proxy)"
echo "- GSM8K: Math word problems (numerical reasoning)"  
echo "- Financial PhraseBank: Financial sentiment analysis"
echo "- FinQA: Financial question answering with tables"
echo "- MMLU: Diverse domain knowledge"
echo ""
echo "To use these datasets in experiments, see:"
echo "  - experiments/run_agent_experiments.py"
echo "  - notebooks/01_ace_demo.ipynb"
