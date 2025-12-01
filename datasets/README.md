# Datasets for Finetuning-Proof Research

This directory contains datasets for researching finetuning-proof properties. Data files are NOT committed to git due to size. Follow the download instructions below.

## Overview

We have identified several categories of datasets relevant to finetuning-proof research:

1. **Dynamic/Temporal Benchmarks**: Continuously updated to avoid contamination
2. **Contamination-Free Benchmarks**: Rewritten versions of popular benchmarks
3. **Robust Mathematical Reasoning**: Symbolic variants that test generalization
4. **Out-of-Distribution Evaluation**: Datasets designed to test distribution shift robustness

## Dataset 1: MMLU-CF (Contamination-Free MMLU)

### Overview
- **Source**: [microsoft/MMLU-CF on HuggingFace](https://huggingface.co/datasets/microsoft/MMLU-CF)
- **Size**: 10,000 questions (test set) + 10,000 questions (validation set)
- **Format**: HuggingFace Dataset (multiple choice questions)
- **Task**: Multi-task language understanding across various academic subjects
- **License**: CDLA-2.0
- **Why Relevant**: Specifically designed to be contamination-free through rewriting; shows significant performance drops (GPT-4o: 88% → 73.4% vs original MMLU)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

# Load MMLU-CF dataset
dataset = load_dataset("microsoft/MMLU-CF")

# Save locally
dataset.save_to_disk("datasets/mmlu_cf")
```

**Alternative (CLI):**
```bash
# Install huggingface-cli if needed
pip install -U "huggingface_hub[cli]"

# Download dataset
huggingface-cli download microsoft/MMLU-CF --repo-type dataset --local-dir datasets/mmlu_cf
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mmlu_cf")

# Or load directly from HuggingFace
from datasets import load_dataset
dataset = load_dataset("microsoft/MMLU-CF")
```

### Sample Data Structure

```json
{
  "question": "What is the primary function of...",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "B",
  "subject": "biology"
}
```

### Notes
- Questions underwent contamination-free processing to ensure they're unlikely to be in training data
- Performance significantly lower than original MMLU, suggesting original was contaminated
- Useful for testing if models truly understand vs. memorize

## Dataset 2: MMLU-Pro

### Overview
- **Source**: [TIGER-Lab/MMLU-Pro on HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- **Size**: 12,000 complex questions across various disciplines
- **Format**: HuggingFace Dataset (multiple choice with 10 options)
- **Task**: Multi-task language understanding with increased difficulty
- **Published**: NeurIPS 2024
- **Why Relevant**: More challenging version of MMLU with 10 options instead of 4; harder to solve through memorization

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

# Load MMLU-Pro dataset
dataset = load_dataset("TIGER-Lab/MMLU-Pro")

# Save locally
dataset.save_to_disk("datasets/mmlu_pro")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mmlu_pro")

# Or load directly
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/MMLU-Pro")
```

### Notes
- Requires chain-of-thought (CoT) reasoning for better results
- 10 answer choices make random guessing less effective
- More robust to finetuning due to increased complexity

## Dataset 3: GSM-Symbolic

### Overview
- **Source**: [apple/GSM-Symbolic on HuggingFace](https://huggingface.co/datasets/apple/GSM-Symbolic)
- **Size**: Dynamically generated from templates (effectively infinite variants)
- **Format**: HuggingFace Dataset (math word problems)
- **Task**: Grade school math reasoning
- **Paper**: arXiv 2410.05229 (ICLR 2025)
- **Why Relevant**: Creates symbolic variants of GSM8K to test if models memorize vs. reason; reveals performance fragility to numerical changes

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

# Load different variants
ds_main = load_dataset("apple/GSM-Symbolic", name="main")
ds_p1 = load_dataset("apple/GSM-Symbolic", name="p1")  # More difficult
ds_p2 = load_dataset("apple/GSM-Symbolic", name="p2")  # Most difficult

# Save locally
ds_main.save_to_disk("datasets/gsm_symbolic_main")
ds_p1.save_to_disk("datasets/gsm_symbolic_p1")
ds_p2.save_to_disk("datasets/gsm_symbolic_p2")
```

### Loading the Dataset

```python
from datasets import load_from_disk

# Load saved datasets
ds_main = load_from_disk("datasets/gsm_symbolic_main")
ds_p1 = load_from_disk("datasets/gsm_symbolic_p1")
ds_p2 = load_from_disk("datasets/gsm_symbolic_p2")
```

### Sample Data

```json
{
  "question": "John has 5 apples. He buys 3 more apples. How many apples does John have now?",
  "answer": "8"
}
```

### Notes
- Templates allow generating infinite variants with different numbers/names
- Performance varies significantly across variants, indicating memorization
- Adding irrelevant clauses causes up to 65% performance drop
- GSM-Symbolic-P2 > P1 > Main in difficulty

## Dataset 4: LiveCodeBench

### Overview
- **Source**: [livecodebench/code_generation_lite on HuggingFace](https://huggingface.co/datasets/livecodebench/code_generation_lite)
- **GitHub**: [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- **Size**:
  - release_v1: 400 problems (May 2023-Mar 2024)
  - release_v2: 511 problems (May 2023-May 2024)
  - release_v3: 612 problems (May 2023-Jul 2024)
  - release_v4: 713 problems (May 2023-Sep 2024)
  - release_v5: 880 problems (May 2023-Jan 2025)
  - release_v6: 1055 problems (May 2023-Apr 2025)
- **Format**: HuggingFace Dataset (coding problems)
- **Task**: Code generation from LeetCode, AtCoder, and CodeForces
- **Why Relevant**: Continuously collects new problems; can evaluate on problems released after model's training cutoff date

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

# Load specific version (use latest for contamination-free evaluation)
dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6")

# Save locally
dataset.save_to_disk("datasets/livecodebench_v6")
```

**Using GitHub:**
```bash
# Clone the full repository with evaluation code
git clone https://github.com/LiveCodeBench/LiveCodeBench.git code/livecodebench
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/livecodebench_v6")
```

### Notes
- Problems are annotated with release dates
- Evaluate models on problems released after training cutoff for contamination-free assessment
- Updated regularly with new contest problems
- Provides holistic evaluation: code generation, test output prediction, and self-repair

## Dataset 5: LiveBench

### Overview
- **Source**: [livebench/coding on HuggingFace](https://huggingface.co/datasets/livebench/coding)
- **GitHub**: [LiveBench/LiveBench](https://github.com/LiveBench/LiveBench)
- **Size**: Updated monthly with new questions
- **Format**: Multiple task types across reasoning, math, coding, data analysis
- **Task**: General LLM evaluation across multiple domains
- **Why Relevant**: Monthly updates based on recent arXiv papers, news, IMDb synopses; designed to avoid contamination

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

# Load LiveBench coding subset
dataset = load_dataset("livebench/coding")

# Save locally
dataset.save_to_disk("datasets/livebench_coding")
```

**Using GitHub:**
```bash
# Clone the full repository
git clone https://github.com/LiveBench/LiveBench.git code/livebench
```

### Notes
- Questions generated from recent sources (arXiv papers, news, movies)
- Updated monthly to prevent contamination
- Contains code from LiveCodeBench and IFEval
- Objective, automated evaluation

## Dataset 6: OpenOOD Benchmarks

### Overview
- **Source**: [OpenOOD GitHub](https://github.com/Jingkang50/OpenOOD)
- **Paper**: arXiv 2210.07242 (NeurIPS 2022), v1.5: arXiv 2306.09301
- **Size**: 9 benchmarks across CIFAR-10, CIFAR-100, ImageNet-200, ImageNet-1K
- **Format**: Image classification datasets
- **Task**: Out-of-distribution detection
- **Why Relevant**: Tests model robustness to distribution shift; relevant for testing if finetuning improves robustness vs. just memorization

### Download Instructions

**Using GitHub:**
```bash
# Clone OpenOOD repository
git clone https://github.com/Jingkang50/OpenOOD.git code/openood

# Follow their data preparation scripts
cd code/openood
# See their documentation for dataset downloads
```

**ID Datasets:**
- CIFAR-10, CIFAR-100: Automatically downloaded via torchvision
- ImageNet-200, ImageNet-1K: Manual download required

**OOD Test Datasets:**
- Near-OOD: SSB-hard, NINCO
- Far-OOD: iNaturalist, Texture, OpenImage-O
- Covariate-Shifted: ImageNet-C, ImageNet-R, ImageNet-v2

### Notes
- Provides unified evaluation framework with 35+ methods
- Includes full-spectrum detection (semantic shift + covariate shift)
- Metrics: AUROC, AUPR, FPR@95
- Well-documented evaluation methodology

## Summary of Dataset Properties

| Dataset | Type | Contamination Resistance | Update Frequency | Size |
|---------|------|-------------------------|------------------|------|
| MMLU-CF | Static, rewritten | High | One-time | 20K questions |
| MMLU-Pro | Static, harder | Medium-High | One-time | 12K questions |
| GSM-Symbolic | Dynamic, templated | High | Generated | Infinite variants |
| LiveCodeBench | Dynamic, temporal | Very High | Continuous | 1055+ problems |
| LiveBench | Dynamic, temporal | Very High | Monthly | Growing |
| OpenOOD | Static, OOD | High | One-time | Multiple benchmarks |

## Evaluation Strategy for Finetuning-Proof Research

To test if a dataset is "finetuning-proof":

1. **Baseline Evaluation**: Test pretrained model on dataset
2. **Finetune on Training Split**: Fine-tune model on dataset's training data
3. **Re-evaluate**: Test finetuned model on test split
4. **Measure Improvement**:
   - Small improvement → More finetuning-proof
   - Large improvement → Less finetuning-proof (likely memorization)

5. **Variant Testing** (for GSM-Symbolic): Test on different symbolic variants to detect memorization vs. reasoning

6. **Temporal Testing** (for LiveCodeBench/LiveBench): Only evaluate on problems released after model's training cutoff

7. **OOD Testing** (for OpenOOD): Test if finetuning improves robustness to distribution shift or just in-distribution memorization

## References

- MMLU-CF: https://arxiv.org/abs/2412.15194 (ACL 2025)
- MMLU-Pro: https://arxiv.org/abs/2406.01574 (NeurIPS 2024)
- GSM-Symbolic: https://arxiv.org/abs/2410.05229 (ICLR 2025)
- LiveCodeBench: https://livecodebench.github.io/
- LiveBench: https://livebench.ai/
- OpenOOD: https://arxiv.org/abs/2210.07242 (NeurIPS 2022)
