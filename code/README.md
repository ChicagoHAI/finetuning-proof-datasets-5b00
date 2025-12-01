# Cloned Repositories

This directory contains code repositories related to finetuning-proof datasets, contamination detection, and robust evaluation benchmarks.

## Repository 1: MMLU-CF

- **URL**: https://github.com/microsoft/MMLU-CF
- **Purpose**: Official implementation and evaluation code for MMLU-CF (Contamination-Free Multi-task Language Understanding Benchmark)
- **Location**: `code/MMLU-CF/`
- **Key Files**:
  - `evaluation/`: Evaluation scripts for running models on MMLU-CF
  - `data/`: Dataset files and processing scripts
  - `README.md`: Setup and usage instructions
- **License**: MIT License
- **Publication**: ACL 2025
- **Notes**: Provides tools to evaluate models on contamination-free MMLU questions; useful for baseline comparisons

## Repository 2: LiveCodeBench

- **URL**: https://github.com/LiveCodeBench/LiveCodeBench
- **Purpose**: Holistic and contamination-free evaluation of LLMs for code generation
- **Location**: `code/LiveCodeBench/`
- **Key Files**:
  - `lcb_runner/`: Main evaluation runner
  - `lcb_runner/benchmarks/`: Benchmark definitions and problem loaders
  - `lcb_runner/evaluation/`: Evaluation metrics and scoring
  - `lcb_runner/prompts/`: Prompt templates for different models
  - `scripts/`: Utility scripts for data processing
- **Key Features**:
  - Continuous collection of new problems from LeetCode, AtCoder, CodeForces
  - Temporal evaluation (can filter by release date)
  - Three task scenarios: code generation, test output prediction, self-repair
- **Usage**:
  ```bash
  cd code/LiveCodeBench
  pip install -e .
  # See README for evaluation instructions
  ```
- **Notes**: Problems annotated with release dates enable contamination-free evaluation by testing only on post-training-cutoff problems

## Repository 3: LiveBench

- **URL**: https://github.com/LiveBench/LiveBench
- **Purpose**: Challenging, contamination-free LLM benchmark across multiple domains
- **Location**: `code/LiveBench/`
- **Key Files**:
  - `livebench/`: Main benchmark package
  - `livebench/process_results/`: Result processing and scoring
  - `livebench/gen_api_answer.py`: Script to generate model answers via API
  - `livebench/gen_ground_truth_judgment.py`: Ground truth generation
- **Key Features**:
  - Monthly updates with new questions
  - Questions based on recent arXiv papers, news, IMDb synopses
  - Objective automated evaluation
  - Multiple task categories: reasoning, math, coding, data analysis
- **Usage**:
  ```bash
  cd code/LiveBench
  pip install -e .
  # See README for running benchmarks
  ```
- **Notes**: Contains code from LiveCodeBench and IFEval; designed to minimize test set contamination

## Repository 4: OpenOOD

- **URL**: https://github.com/Jingkang50/OpenOOD
- **Purpose**: Benchmarking Generalized Out-of-Distribution Detection
- **Location**: `code/OpenOOD/`
- **Key Files**:
  - `openood/`: Main package
  - `openood/datasets/`: Dataset loaders and preprocessors
  - `openood/evaluators/`: Evaluation metrics (AUROC, AUPR, FPR@95)
  - `openood/networks/`: Neural network architectures
  - `openood/postprocessors/`: OOD detection methods (35+ implemented)
  - `scripts/`: Training and evaluation scripts
  - `configs/`: Configuration files for different benchmarks
- **Key Features**:
  - 9 benchmarks (CIFAR-10/100, ImageNet-200/1K with various OOD test sets)
  - 35+ OOD detection methods implemented
  - Full-spectrum detection (semantic shift + covariate shift)
  - Standardized evaluation protocol
- **Benchmarks Included**:
  - Near-OOD: SSB-hard, NINCO
  - Far-OOD: iNaturalist, Texture, OpenImage-O
  - Covariate-Shifted: ImageNet-C, ImageNet-R, ImageNet-v2
- **Usage**:
  ```bash
  cd code/OpenOOD
  pip install -r requirements.txt
  # See documentation for dataset preparation
  bash scripts/download/download_imgnet.sh  # ImageNet datasets
  bash scripts/download/download_ood_data.sh  # OOD test sets
  ```
- **Publications**:
  - v1.0: NeurIPS 2022 (Datasets and Benchmarks Track)
  - v1.5: DMLR 2024 (Journal of Data-centric Machine Learning Research)
- **Notes**: Comprehensive framework for testing model robustness to distribution shift; relevant for understanding if finetuning improves generalization or just memorization

## Repository 5: awesome-data-contamination

- **URL**: https://github.com/lyy1994/awesome-data-contamination
- **Purpose**: Curated list of papers on data contamination for Large Language Models evaluation
- **Location**: `code/awesome-data-contamination/`
- **Key Files**:
  - `README.md`: Comprehensive list of papers organized by topic
- **Categories Covered**:
  - Contamination detection methods
  - Contamination in pre-training
  - Contamination in fine-tuning
  - Benchmark contamination studies
  - Mitigation strategies
- **Notes**: Excellent resource for literature review; regularly updated with new papers

## Summary of Use Cases

| Repository | Primary Use | Evaluation Type | Contamination Strategy |
|-----------|-------------|-----------------|----------------------|
| MMLU-CF | Language understanding | Static, rewritten | Question rewriting |
| LiveCodeBench | Code generation | Dynamic, temporal | Release date filtering |
| LiveBench | General reasoning | Dynamic, temporal | Monthly fresh questions |
| OpenOOD | OOD detection | Static, distribution shift | Domain separation |
| awesome-data-contamination | Literature review | N/A | N/A |

## Getting Started

### General Setup

Most repositories require Python 3.8+ and PyTorch. Install dependencies for each:

```bash
# MMLU-CF
cd code/MMLU-CF && pip install -r requirements.txt

# LiveCodeBench
cd code/LiveCodeBench && pip install -e .

# LiveBench
cd code/LiveBench && pip install -e .

# OpenOOD
cd code/OpenOOD && pip install -r requirements.txt
```

### Typical Workflow

1. **Prepare Dataset**: Follow dataset-specific instructions in each repository
2. **Run Baseline**: Evaluate pretrained model on test set
3. **Fine-tune**: Fine-tune model on training data
4. **Re-evaluate**: Test finetuned model on test set
5. **Compare**: Measure performance change to assess "finetuning-proof" properties

### Key Metrics

- **Standard Benchmarks**: Accuracy, F1, exact match
- **OOD Detection**: AUROC, AUPR, FPR@95
- **Robustness**: Performance variance across distribution shifts
- **Contamination Signal**: Performance drop on rewritten/fresh vs. original questions

## Integration Notes

These repositories can be integrated into an experiment pipeline to:

1. Test baseline model performance
2. Fine-tune on various datasets
3. Evaluate on both original and contamination-free variants
4. Measure robustness to distribution shifts
5. Quantify "finetuning-proof" properties

The experiment runner can use these codebases as building blocks for systematic evaluation of finetuning resistance across different dataset types.
