# Downloaded Papers

This directory contains research papers related to finetuning-proof datasets, benchmark saturation, data contamination, and robust evaluation.

## Papers on Data Contamination and Fine-tuning

### 1. Sensitivity of Small Language Models to Fine-tuning Data Contamination
- **File**: `2511.06763_sensitivity_finetuning_contamination.pdf`
- **arXiv**: 2511.06763
- **Year**: 2025 (November)
- **Key Focus**: Systematically investigates contamination sensitivity of 23 small language models (270M-4B parameters) during instruction tuning
- **Why Relevant**: Directly addresses how fine-tuning on contaminated data affects model performance, revealing catastrophic degradation with syntactic transformations

### 2. DICE: Detecting In-distribution Contamination in LLM's Fine-tuning Phase for Math Reasoning
- **File**: `2406.04197_DICE_contamination_detection.pdf`
- **arXiv**: 2406.04197
- **Year**: 2024 (June)
- **Key Focus**: Novel method to detect in-distribution contamination during fine-tuning by analyzing layer-specific activation patterns
- **Why Relevant**: Provides methodology to identify when fine-tuning datasets overlap semantically with test data, critical for understanding "finetuning-proof" properties

### 3. Comprehensive Survey of Contamination Detection Methods in Large Language Models
- **File**: `2404.00699_contamination_detection_survey.pdf`
- **arXiv**: 2404.00699
- **Year**: 2024 (April)
- **Key Focus**: Comprehensive survey of contamination detection methods across pre-training and fine-tuning phases
- **Why Relevant**: Provides overview of how contamination affects evaluation and what methods exist to detect it

## Papers on Benchmark Robustness and Limitations

### 4. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models
- **File**: `2410.05229_GSM_Symbolic_reasoning_limitations.pdf`
- **arXiv**: 2410.05229
- **Year**: 2024 (October), Published at ICLR 2025
- **Key Focus**: Creates symbolic templates to generate diverse question variants, revealing LLMs' fragility to minor changes
- **Why Relevant**: Demonstrates that GSM8K performance may reflect memorization rather than true reasoning - directly relevant to whether datasets become easier after fine-tuning due to memorization

### 5. Language Model Developers Should Report Train-Test Overlap
- **File**: `2410.08385_train_test_overlap.pdf`
- **arXiv**: 2410.08385
- **Year**: 2024 (October)
- **Key Focus**: Analyzes transparency in train-test overlap reporting across 30 model developers
- **Why Relevant**: Establishes the prevalence and impact of train-test overlap, which is central to understanding why datasets become easier after fine-tuning

### 6. Rethinking Benchmark and Contamination for Language Models with Rephrased Samples
- **File**: `2311.04850_rethinking_benchmark_contamination.pdf`
- **arXiv**: 2311.04850
- **Year**: 2023 (November)
- **Key Focus**: Proposes using rephrased samples to evaluate contamination effects
- **Why Relevant**: Offers methodology for testing dataset robustness to fine-tuning through paraphrasing

## Papers on Contamination-Free Evaluation

### 7. MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark
- **File**: `2412.15194_MMLU_CF_contamination_free.pdf`
- **arXiv**: 2412.15194
- **Year**: 2024 (December), Accepted at ACL 2025
- **Key Focus**: Creates contamination-free version of MMLU through question rewriting
- **Why Relevant**: Demonstrates approach to creating "finetuning-proof" datasets by ensuring questions aren't in training data; shows significant performance drops (GPT-4o: 88% → 73.4%)

### 8. Benchmarking Large Language Models Under Data Contamination: A Survey from Static to Dynamic Evaluation
- **File**: `2502.17521_dynamic_evaluation_survey.pdf`
- **arXiv**: 2502.17521
- **Year**: 2025 (February)
- **Key Focus**: Comprehensive survey of dynamic benchmarking approaches to combat contamination
- **Why Relevant**: Reviews temporal cutoff, rule-based generation, and LLM-based generation methods for creating robust evaluation datasets

## Papers on Adversarial Robustness

### 9. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation
- **File**: `2403.13322_DD_RobustBench.pdf`
- **arXiv**: 2403.13322
- **Year**: 2024 (March)
- **Key Focus**: Comprehensive benchmark evaluating adversarial robustness of distilled datasets
- **Why Relevant**: Explores how dataset properties affect robustness to fine-tuning and adversarial attacks

## Summary

These papers collectively address:
1. **Contamination Detection**: Methods to identify train-test overlap during fine-tuning
2. **Benchmark Saturation**: Evidence that popular benchmarks become too easy due to contamination
3. **Robust Evaluation**: Approaches to create contamination-free, dynamic benchmarks
4. **Reasoning vs. Memorization**: Evidence that fine-tuning often leads to memorization rather than generalization

The research suggests that most existing benchmarks are NOT finetuning-proof, but several approaches exist to create more robust evaluation datasets.
