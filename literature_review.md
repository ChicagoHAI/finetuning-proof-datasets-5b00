# Literature Review: Finetuning-Proof Datasets

## Executive Summary

This literature review addresses the research question: **"Are there any finetuning-proof datasets currently?"**

**Key Finding**: Most existing benchmarks are NOT truly finetuning-proof. Fine-tuning often leads to inflated performance through memorization rather than genuine reasoning improvement. However, several promising approaches have emerged to create more robust evaluation datasets:

1. **Contamination-Free Rewriting** (MMLU-CF): Rewriting existing benchmarks to avoid training data overlap
2. **Dynamic/Temporal Benchmarks** (LiveCodeBench, LiveBench): Continuously updated datasets with release date annotations
3. **Symbolic Variants** (GSM-Symbolic): Template-based generation to test generalization vs. memorization
4. **Out-of-Distribution Testing** (OpenOOD): Evaluating robustness to distribution shifts

The research suggests that "finetuning-proof" is not binary but a spectrum, with some datasets more resistant than others. True resistance requires continuous updates, careful contamination detection, and testing of genuine reasoning rather than pattern matching.

## Research Area Overview

### The Problem: Benchmark Contamination and Saturation

Modern large language models (LLMs) are trained on massive web-scale datasets, inevitably including benchmark questions either directly or in paraphrased form. This leads to:

- **Train-test overlap**: Test data appearing in training sets (Mirzadeh et al., 2024)
- **Benchmark saturation**: Models achieving near-perfect scores, eliminating differentiation (MMLU: 88-91% clustering)
- **Memorization vs. reasoning**: Models replicating training patterns rather than true understanding (GSM-Symbolic findings)
- **Performance inflation**: Metrics reflecting exposure rather than capability

### The Research Question

This review investigates which datasets are most resistant to fine-tuning and whether any are truly "finetuning-proof" - maintaining difficulty and requiring genuine generalization even after fine-tuning exposure.

## Key Papers

### 1. Sensitivity of Small Language Models to Fine-tuning Data Contamination

**Citation**: arXiv:2511.06763 (November 2025)
**Authors**: Multiple authors
**Key Contribution**: Systematic investigation of contamination sensitivity across 23 small language models (270M-4B parameters)

**Methodology**:
- Tested sensitivity to syntactic transformations (character and word reversal)
- Tested sensitivity to semantic transformations (irrelevant and counterfactual responses)
- Measured performance degradation during instruction tuning

**Key Results**:
- **Catastrophic failure with syntactic contamination**: Character reversal causes near-complete failure across all models regardless of size or family
- **Threshold behavior with semantic contamination**: Distinct thresholds where performance collapses
- **Size-independent vulnerability**: Even 4B parameter models highly sensitive to contamination

**Relevance**: Demonstrates that fine-tuning on contaminated data has severe, predictable effects on model performance, providing quantitative evidence for contamination detection.

**Datasets Used**: Various instruction-tuning datasets with synthetic contamination injected

**Limitations**: Focus on small models; unclear if findings fully generalize to frontier models

### 2. DICE: Detecting In-distribution Contamination in LLM's Fine-tuning Phase for Math Reasoning

**Citation**: arXiv:2406.04197 (June 2024)
**Authors**: Multiple authors
**Key Contribution**: Novel method to detect in-distribution contamination during fine-tuning

**Methodology**:
- Identifies most sensitive layer to contamination through layer-wise analysis
- Trains contamination classifier based on internal states of sensitive layer
- Tests on math reasoning datasets (GSM8K, MATH)

**Key Results**:
- **High detection accuracy**: Successfully identifies in-distribution contamination across various LLMs
- **Layer-specific sensitivity**: Different layers respond differently to contamination
- **Semantic overlap detection**: Can detect contamination even without exact text matching

**Relevance**: Provides practical tool for identifying when fine-tuning datasets overlap with test data, critical for assessing true "finetuning-proof" properties.

**Code Available**: Yes (methods can be implemented)

**Baselines**: Traditional perplexity-based methods (which fail for in-distribution contamination)

### 3. Comprehensive Survey of Contamination Detection Methods

**Citation**: arXiv:2404.00699 (April 2024)
**Authors**: Multiple authors
**Key Contribution**: Comprehensive survey of contamination detection across pre-training and fine-tuning

**Coverage**:
- Pre-training phase leakage mechanisms
- Fine-tuning biases and overlap
- Cross-modal contamination
- Detection methodologies

**Key Findings**:
- **Pervasive problem**: Contamination affects most major benchmarks
- **Multiple pathways**: Contamination occurs through web scraping, dataset leakage, synthetic data generation
- **Detection challenges**: No single method works for all contamination types

**Relevance**: Provides taxonomic understanding of contamination types and detection approaches.

### 4. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models

**Citation**: arXiv:2410.05229 (October 2024, ICLR 2025)
**Authors**: Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar (Apple)
**Key Contribution**: Creates symbolic templates to generate diverse math question variants, revealing fragility of LLM reasoning

**Methodology**:
- Generated GSM-Symbolic from symbolic templates
- Created multiple instantiations of same logical structure with different numbers/names
- Tested all state-of-the-art models on variants
- Added irrelevant clauses to test robustness

**Key Results**:
- **Performance variance**: Noticeable variance when only numerical values change (suggests memorization)
- **Fragility with complexity**: Performance significantly deteriorates as clauses increase
- **Sensitivity to irrelevant information**: Up to 65% performance drop when adding single irrelevant clause
- **Limited true reasoning**: Models replicate reasoning steps from training data rather than performing genuine logic

**Relevance**: Demonstrates that GSM8K performance largely reflects memorization, not reasoning. Symbolic variants provide "finetuning-proof" evaluation by testing generalization.

**Datasets**: GSM8K (original), GSM-Symbolic, GSM-Symbolic-P1, GSM-Symbolic-P2 (increasing difficulty)

**Code Available**: Yes, with HuggingFace dataset (apple/GSM-Symbolic)

**Impact**: Questions reliability of mathematical reasoning benchmarks; shows path toward more robust evaluation

### 5. Language Model Developers Should Report Train-Test Overlap

**Citation**: arXiv:2410.08385 (October 2024)
**Authors**: Multiple authors
**Key Contribution**: Analyzes transparency in train-test overlap reporting across 30 model developers

**Methodology**:
- Surveyed 30 model developers for train-test overlap statistics
- Examined public documentation and technical reports
- Quantified overlap in popular datasets (C4, RealNews)

**Key Results**:
- **Limited transparency**: Only 9 of 30 developers report train-test overlap
- **Significant overlap discovered**: 4.6% of C4 validation set, 14.4% of RealNews validation set had training duplicates
- **Inflated metrics**: Models better at memorizing show unfairly inflated evaluation scores
- **No standardized reporting**: Different developers use different overlap definitions

**Recommendations**:
- Developers should publish train-test overlap statistics
- Release training data under open-source licenses (4 developers currently do this)
- Publish methodology and statistics (5 developers currently do this)

**Relevance**: Establishes baseline for contamination prevalence; demonstrates need for "finetuning-proof" benchmarks

**Evaluation Metrics**: Percentage overlap, duplicate detection methods

### 6. Rethinking Benchmark and Contamination for Language Models with Rephrased Samples

**Citation**: arXiv:2311.04850 (November 2023)
**Authors**: Multiple authors
**Key Contribution**: Proposes using rephrased samples to evaluate contamination effects

**Methodology**:
- Generate paraphrased versions of benchmark questions
- Compare model performance on original vs. paraphrased
- Performance drop indicates potential contamination/memorization

**Key Results**:
- **Performance gaps reveal contamination**: Large gaps between original and paraphrased indicate memorization
- **Viable mitigation strategy**: Paraphrasing can create contamination-resistant variants
- **Not perfect**: Sophisticated models can still generalize across paraphrases

**Relevance**: Provides practical approach for creating "finetuning-proof" variants through linguistic variation

**Limitations**: Paraphrasing may change difficulty; semantic equivalence hard to guarantee

### 7. MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark

**Citation**: arXiv:2412.15194 (December 2024, ACL 2025)
**Authors**: Microsoft Research
**Key Contribution**: Creates contamination-free version of MMLU through systematic question rewriting

**Methodology**:
- MCQ Collection from multiple sources
- MCQ Cleaning and quality control
- Difficulty sampling to match original MMLU
- LLM checking (GPT-4o, Gemini, Claude)
- **Contamination-Free Processing**: Five-step process with three rewriting rules

**Key Results**:
- **Significant performance drops**: GPT-4o: 88.0% (MMLU) → 73.4% (MMLU-CF)
- **Reveals true capabilities**: 14.6 percentage point drop indicates original MMLU contamination
- **Challenging benchmark**: Top models cluster 70-75%, providing better differentiation
- **Maintained difficulty distribution**: MMLU-CF difficulty profile similar to original

**Dataset Size**: 10,000 questions (test), 10,000 questions (validation)

**License**: CDLA-2.0

**Relevance**: Provides concrete example of "finetuning-proof" benchmark through contamination elimination. Shows that contamination-free evaluation reveals significant performance gaps.

**Code Available**: Yes (https://github.com/microsoft/MMLU-CF)

**Evaluation Metrics**: Accuracy across subject areas

**Impact**: Sets new standard for contamination-free evaluation; demonstrates feasibility of rewriting approach

### 8. Benchmarking Large Language Models Under Data Contamination: A Survey from Static to Dynamic Evaluation

**Citation**: arXiv:2502.17521 (February 2025)
**Authors**: Multiple authors
**Key Contribution**: Comprehensive survey of dynamic benchmarking approaches

**Taxonomy of Dynamic Benchmarking**:
1. **Temporal cutoff**: Using data after model's knowledge cutoff date
2. **Rule-based generation**: Algorithmic question generation following templates
3. **LLM-based generation**: Using LLMs to create new evaluation questions
4. **Hybrid approaches**: Combining multiple strategies

**Key Benchmarks Reviewed**:
- LiveCodeBench (temporal cutoff for code)
- LiveBench (monthly updates across domains)
- LLMEval-3 (220K+ private question bank)
- LiveXiv (evolving VLM benchmark from arXiv papers)

**Key Findings**:
- **Static benchmarks vulnerable**: All major static benchmarks show contamination signals
- **Dynamic solutions emerging**: Temporal and generative approaches show promise
- **Trade-offs exist**: Dynamic benchmarks face challenges in consistency, cost, quality control

**Relevance**: Provides roadmap for creating and maintaining "finetuning-proof" benchmarks through dynamic approaches

**Recommendations**: Combination of temporal cutoffs and generation methods for maximum robustness

### 9. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation

**Citation**: arXiv:2403.13322 (March 2024)
**Authors**: Multiple authors
**Key Contribution**: Comprehensive benchmark evaluating adversarial robustness of distilled datasets

**Methodology**:
- Evaluates dataset distillation methods (TESLA, SRe2L, DM, IDM, BACON)
- Tests adversarial attacks (FGSM, PGD, C&W)
- Across datasets (CIFAR-10/100, TinyImageNet, ImageNet-1K)

**Key Results**:
- **Distillation reduces robustness**: Distilled datasets often less robust than original
- **Method-dependent**: Some distillation methods preserve robustness better
- **Attack transferability**: Models trained on distilled data vulnerable to similar attacks

**Relevance**: Shows that dataset properties (like distillation) affect robustness; relevant for understanding what makes datasets "finetuning-proof"

**Datasets**: CIFAR-10, CIFAR-100, TinyImageNet, ImageNet-1K

**Metrics**: Accuracy under various attack strengths

## Common Methodologies for Creating Finetuning-Proof Datasets

### 1. Temporal Cutoff Approach

**Mechanism**: Only use data created after model's training cutoff date

**Examples**:
- LiveCodeBench: Annotates problems with release dates; evaluate on post-cutoff problems
- LiveBench: Monthly updates with questions from recent arXiv papers, news, movies

**Advantages**:
- Guaranteed no direct training exposure
- Automatically contamination-free
- Can be continuously updated

**Limitations**:
- Requires continuous data collection infrastructure
- May introduce temporal biases
- Quality control challenges with new data

**Effectiveness**: Very high for preventing direct memorization; best current approach

### 2. Question Rewriting/Paraphrasing

**Mechanism**: Systematically rewrite existing benchmark questions to avoid training data overlap

**Examples**:
- MMLU-CF: Five-step contamination-free processing with three rewriting rules
- Rephrased sampling approaches

**Advantages**:
- Can retrofit existing benchmarks
- Maintains similar difficulty distribution
- One-time effort

**Limitations**:
- Expensive (requires human or LLM review)
- Semantic equivalence hard to guarantee
- May inadvertently change difficulty
- Static (can become contaminated over time)

**Effectiveness**: High initially; degrades as rewritten versions enter training data

### 3. Symbolic/Template-Based Generation

**Mechanism**: Generate infinite variants from templates, testing generalization rather than memorization

**Examples**:
- GSM-Symbolic: Templates with variable instantiation for math problems
- Rule-based generation systems

**Advantages**:
- Infinite test examples possible
- Can isolate reasoning from memorization
- Controllable difficulty

**Limitations**:
- Limited to structured domains (math, code, logic)
- Template design requires expertise
- May not capture full task complexity

**Effectiveness**: Very high for detecting memorization vs. reasoning; best for structured tasks

### 4. Out-of-Distribution Testing

**Mechanism**: Evaluate on data from different distributions than training

**Examples**:
- OpenOOD: Near-OOD and Far-OOD test sets
- Domain shift benchmarks (ImageNet-C, ImageNet-R)

**Advantages**:
- Tests true generalization
- Can't be solved through memorization
- Reveals model robustness

**Limitations**:
- Doesn't test same capability as in-distribution
- May conflate generalization with domain shift robustness
- Requires careful OOD data selection

**Effectiveness**: High for testing generalization; complementary to in-distribution evaluation

## Standard Baselines

### For Language Understanding

**Models Commonly Evaluated**:
- GPT-4, GPT-4o (OpenAI)
- Claude 3, Claude 3.5 (Anthropic)
- Gemini, Gemini Pro (Google)
- Llama 2, Llama 3 (Meta)
- Mistral, Mixtral (Mistral AI)

**Typical Performance Ranges**:
- MMLU: 85-91% (top models, likely contaminated)
- MMLU-CF: 70-75% (contamination-free)
- Performance drop: 10-15 percentage points reveals contamination

### For Mathematical Reasoning

**Models**:
- Same frontier LLMs plus specialized math models
- Code-trained models often perform better

**Typical Performance**:
- GSM8K: 90-95% (top models, likely memorized)
- GSM-Symbolic: Significant variance and degradation
- Irrelevant clause sensitivity: Up to 65% drop

### For Code Generation

**Models**:
- Code-specialized: GPT-4 (code mode), Claude 3.5 Sonnet, Codex
- General models with code training

**Typical Performance**:
- Static benchmarks: High performance (likely contaminated)
- LiveCodeBench: Lower performance on post-cutoff problems
- Temporal gap indicates memorization

## Evaluation Metrics

### Contamination Detection Metrics

1. **Performance Gap**: Difference between original and contamination-free variants
   - Large gap (>10 percentage points) indicates contamination
   - MMLU vs. MMLU-CF: 14.6 points for GPT-4o

2. **Variance Across Variants**: For symbolic/templated datasets
   - High variance suggests memorization rather than reasoning
   - GSM-Symbolic shows significant variance within same logical structure

3. **Temporal Performance Degradation**: For dynamic benchmarks
   - Performance drop on newer vs. older problems
   - Indicates training data recency bias

4. **Sensitivity to Perturbations**:
   - Performance drop when adding irrelevant information
   - GSM-Symbolic: Up to 65% drop with single irrelevant clause
   - Character/word reversal: Near-complete failure

### Standard Evaluation Metrics

1. **Accuracy**: Correct answers / total questions
   - Primary metric for most benchmarks
   - Simple, interpretable

2. **F1 Score**: Harmonic mean of precision and recall
   - Used when class imbalance exists

3. **Exact Match**: Exact string match with reference answer
   - Common for generation tasks

4. **Pass@k**: For code generation
   - Percentage of problems solved with k attempts

5. **OOD Detection Metrics**:
   - **AUROC**: Area under ROC curve (discrimination ability)
   - **AUPR**: Area under precision-recall curve
   - **FPR@95**: False positive rate at 95% true positive rate

## Datasets in the Literature

### Contamination-Resistant Datasets

| Dataset | Type | Resistance Mechanism | Size | Domain |
|---------|------|---------------------|------|--------|
| MMLU-CF | Static, rewritten | Question rewriting | 20K questions | Multi-task language understanding |
| MMLU-Pro | Static, harder | Increased difficulty (10 options) | 12K questions | Multi-task language understanding |
| GSM-Symbolic | Dynamic, templated | Symbolic generation | Infinite variants | Math reasoning |
| LiveCodeBench | Dynamic, temporal | Release date filtering | 1055+ problems | Code generation |
| LiveBench | Dynamic, temporal | Monthly updates | Growing | Multi-domain reasoning |
| OpenOOD | Static, OOD | Distribution shift | Multiple benchmarks | Image classification |

### Contaminated/Saturated Datasets (Use with Caution)

| Dataset | Evidence of Contamination | Alternative |
|---------|---------------------------|-------------|
| MMLU | 88-91% clustering, 14.6 point gap with MMLU-CF | Use MMLU-CF or MMLU-Pro |
| GSM8K | 90-95% performance, high variance on symbolic variants | Use GSM-Symbolic |
| HumanEval | Static code benchmark, likely in training data | Use LiveCodeBench |
| GLUE | Saturated, models near human performance | Use dynamic alternatives |

## Gaps and Opportunities

### Current Gaps

1. **Limited Temporal Benchmarks**: Only a few domains (code, general reasoning) have temporal benchmarks
   - **Opportunity**: Develop temporal benchmarks for specialized domains (medicine, law, science)

2. **Expensive Maintenance**: Dynamic benchmarks require continuous curation
   - **Opportunity**: Automated generation and quality control systems

3. **No Universal Solution**: Different approaches work for different domains
   - **Opportunity**: Develop domain-agnostic contamination detection and mitigation

4. **Small Model Studies Limited**: Most contamination research focuses on large models
   - **Opportunity**: Study contamination effects across model scales systematically

5. **Cross-Modal Contamination**: Limited research on multi-modal contamination
   - **Opportunity**: Investigate vision-language, audio-text contamination

### Emerging Research Directions

1. **Adversarial Contamination Detection**: Using adversarial examples to detect memorization
2. **Watermarking Benchmarks**: Embedding detectable signals in test data
3. **Federated Evaluation**: Private test sets maintained by third parties
4. **Continuous Benchmarking**: Real-time benchmark updates and evaluation

## Recommendations for Our Experiment

### Recommended Datasets

**Priority 1 - High Contamination Resistance**:
1. **LiveCodeBench (v6)**: Best temporal benchmark, continuous updates
2. **MMLU-CF**: Best language understanding benchmark, proven contamination-free
3. **GSM-Symbolic**: Best for testing reasoning vs. memorization

**Priority 2 - Good Contamination Resistance**:
4. **MMLU-Pro**: Harder variant of MMLU, less saturated
5. **LiveBench**: Multi-domain temporal benchmark

**Priority 3 - Complementary Evaluation**:
6. **OpenOOD**: Test robustness to distribution shift

### Recommended Baselines

**For Comparison**:
1. **GPT-4 / GPT-4o**: Industry standard, well-documented
2. **Claude 3.5 Sonnet**: Strong general performance
3. **Llama 3 variants**: Open-source baseline
4. **Smaller models (1-7B)**: Test scale effects on contamination resistance

**Evaluation Protocol**:
1. Test on original contaminated benchmark (if available)
2. Test on contamination-free variant
3. Calculate performance gap
4. For symbolic: Test variance across multiple variants
5. For temporal: Only use post-training-cutoff data

### Recommended Metrics

**Primary Metrics**:
1. **Accuracy** (or task-specific metric)
2. **Performance Gap** (original vs. contamination-free)
3. **Variance Across Variants** (for symbolic datasets)

**Secondary Metrics**:
4. **Temporal Performance Trend** (for dynamic benchmarks)
5. **Perturbation Sensitivity** (irrelevant information robustness)
6. **OOD Robustness** (for distribution shift testing)

### Methodological Considerations

**Critical Design Choices**:

1. **Test Multiple Dataset Types**: No single dataset is perfectly finetuning-proof
   - Use combination of temporal, rewritten, and symbolic approaches
   - Validates findings across methodologies

2. **Document Training Cutoffs**: Essential for temporal evaluation
   - Record exact training data cutoff dates
   - Only evaluate on post-cutoff data for temporal benchmarks

3. **Measure Memorization Signals**:
   - Test on multiple instantiations of same logical structure
   - Add perturbations (irrelevant clauses, paraphrasing)
   - Compare performance on variants

4. **Baseline Comparisons**:
   - Always compare to published results on original benchmarks
   - Calculate performance gaps to quantify contamination
   - Use multiple baseline models to isolate contamination effects

5. **Report Transparency**:
   - Document any potential train-test overlap
   - Report all performance metrics, not just best results
   - Share failure cases and variance

**Statistical Rigor**:
- Run multiple random seeds
- Report confidence intervals
- Test statistical significance of performance gaps
- Use bootstrapping for small datasets

## Conclusion

### Are There Finetuning-Proof Datasets?

**Answer**: No dataset is perfectly finetuning-proof, but several approaches show strong resistance:

1. **Most Resistant**: Temporal benchmarks (LiveCodeBench, LiveBench)
   - Continuously updated
   - Guaranteed post-training data
   - Requires maintenance infrastructure

2. **Highly Resistant**: Contamination-free rewriting (MMLU-CF)
   - Proven 14.6 percentage point gap vs. original
   - Static but effective initially
   - Will require updates as rewritten versions enter training data

3. **Effective for Detection**: Symbolic variants (GSM-Symbolic)
   - Reveals memorization through variance
   - Infinite test examples
   - Limited to structured domains

4. **Complementary**: OOD testing (OpenOOD)
   - Tests different capability (generalization vs. memorization)
   - High resistance to finetuning on in-distribution data
   - May not reflect same skills

### Key Insights

1. **Contamination is Pervasive**: Most popular benchmarks (MMLU, GSM8K, HumanEval) show strong contamination signals

2. **Finetuning ≠ Generalization**: Fine-tuning often improves benchmark performance through memorization, not true capability improvement

3. **Dynamic > Static**: Temporal approaches most robust; static benchmarks inevitably contaminated over time

4. **Multiple Signals Needed**: No single metric or dataset perfectly captures "finetuning-proof" properties; use multiple approaches

5. **Transparency Critical**: Developers must report train-test overlap and training data cutoffs

### Practical Implications

**For Model Developers**:
- Report train-test overlap statistics
- Evaluate on contamination-free benchmarks
- Use temporal cutoffs for honest evaluation

**For Researchers**:
- Prioritize temporal and contamination-free benchmarks
- Report performance on multiple dataset variants
- Document all potential contamination sources

**For This Research Project**:
- Focus on LiveCodeBench, MMLU-CF, and GSM-Symbolic
- Measure performance gaps as primary signal
- Compare finetuning improvements on contaminated vs. clean benchmarks
- Quantify "finetuning-proof" as spectrum, not binary classification

### Future Directions

The field is rapidly evolving toward:
1. Continuous, automated benchmark generation
2. Federated private test sets
3. Standardized contamination reporting
4. Multi-modal contamination detection
5. Adversarial contamination testing

The most "finetuning-proof" future benchmark will likely combine temporal updates, symbolic generation, and sophisticated contamination detection - requiring significant infrastructure investment but providing trustworthy evaluation.

## References

1. Mirzadeh et al. (2025). Sensitivity of Small Language Models to Fine-tuning Data Contamination. arXiv:2511.06763
2. DICE Team (2024). Detecting In-distribution Contamination in LLM's Fine-tuning Phase. arXiv:2406.04197
3. Survey Authors (2024). Comprehensive Survey of Contamination Detection Methods. arXiv:2404.00699
4. Mirzadeh et al. (2024). GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning. arXiv:2410.05229 (ICLR 2025)
5. Multiple Authors (2024). Language Model Developers Should Report Train-Test Overlap. arXiv:2410.08385
6. Authors (2023). Rethinking Benchmark and Contamination with Rephrased Samples. arXiv:2311.04850
7. Microsoft Research (2024). MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark. arXiv:2412.15194 (ACL 2025)
8. Survey Authors (2025). Benchmarking LLMs Under Data Contamination: Static to Dynamic. arXiv:2502.17521
9. Authors (2024). DD-RobustBench: Adversarial Robustness for Dataset Distillation. arXiv:2403.13322
10. LiveCodeBench Team. LiveCodeBench: Holistic and Contamination Free Evaluation. https://livecodebench.github.io/
11. LiveBench Team. LiveBench: A Challenging, Contamination-Free Benchmark. https://livebench.ai/
12. Jingkang et al. (2022). OpenOOD: Benchmarking Generalized OOD Detection. arXiv:2210.07242 (NeurIPS 2022)
