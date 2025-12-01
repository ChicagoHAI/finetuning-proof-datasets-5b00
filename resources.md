# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project investigating **"Are there any finetuning-proof datasets currently?"**

The research reveals that while no dataset is perfectly finetuning-proof, several promising approaches exist:
- **Temporal benchmarks** (LiveCodeBench, LiveBench): Highest resistance through continuous updates
- **Contamination-free rewriting** (MMLU-CF): Proven effective with 14.6 percentage point drops
- **Symbolic generation** (GSM-Symbolic): Detects memorization through variance testing
- **Out-of-distribution testing** (OpenOOD): Complementary robustness evaluation

## Papers

Total papers downloaded: **9**

| Title | Authors | Year | File | ArXiv | Key Contribution |
|-------|---------|------|------|-------|------------------|
| Sensitivity of Small Language Models to Fine-tuning Data Contamination | Multiple Authors | 2025 | 2511.06763_sensitivity_finetuning_contamination.pdf | 2511.06763 | Systematic study showing catastrophic failure with syntactic contamination |
| DICE: Detecting In-distribution Contamination | Multiple Authors | 2024 | 2406.04197_DICE_contamination_detection.pdf | 2406.04197 | Novel layer-based method for detecting semantic contamination during fine-tuning |
| Survey of Contamination Detection Methods | Multiple Authors | 2024 | 2404.00699_contamination_detection_survey.pdf | 2404.00699 | Comprehensive overview of contamination types and detection approaches |
| GSM-Symbolic: Understanding Limitations of Mathematical Reasoning | Mirzadeh et al. (Apple) | 2024 | 2410.05229_GSM_Symbolic_reasoning_limitations.pdf | 2410.05229 | Reveals memorization vs. reasoning through symbolic variants; up to 65% drops |
| Language Model Developers Should Report Train-Test Overlap | Multiple Authors | 2024 | 2410.08385_train_test_overlap.pdf | 2410.08385 | Documents lack of transparency; only 9/30 developers report overlap |
| Rethinking Benchmark and Contamination with Rephrased Samples | Multiple Authors | 2023 | 2311.04850_rethinking_benchmark_contamination.pdf | 2311.04850 | Proposes paraphrasing methodology for contamination-resistant evaluation |
| MMLU-CF: Contamination-free Multi-task Language Understanding | Microsoft Research | 2024 | 2412.15194_MMLU_CF_contamination_free.pdf | 2412.15194 | Creates contamination-free MMLU; GPT-4o drops from 88% to 73.4% |
| Benchmarking LLMs Under Data Contamination: Static to Dynamic | Multiple Authors | 2025 | 2502.17521_dynamic_evaluation_survey.pdf | 2502.17521 | Survey of dynamic benchmarking approaches: temporal, rule-based, LLM-generated |
| DD-RobustBench: Adversarial Robustness for Dataset Distillation | Multiple Authors | 2024 | 2403.13322_DD_RobustBench.pdf | 2403.13322 | Comprehensive benchmark for evaluating robustness of distilled datasets |

See `papers/README.md` for detailed descriptions and relevance explanations.

## Datasets

Total datasets identified: **6 primary + multiple OOD variants**

| Name | Source | Size | Task | Download | Status |
|------|--------|------|------|----------|--------|
| MMLU-CF | HuggingFace (microsoft/MMLU-CF) | 20K questions | Multi-task language understanding | `load_dataset("microsoft/MMLU-CF")` | Ready to download |
| MMLU-Pro | HuggingFace (TIGER-Lab/MMLU-Pro) | 12K questions | Multi-task language understanding (harder) | `load_dataset("TIGER-Lab/MMLU-Pro")` | Ready to download |
| GSM-Symbolic | HuggingFace (apple/GSM-Symbolic) | Infinite variants | Math reasoning | `load_dataset("apple/GSM-Symbolic")` | Ready to download |
| LiveCodeBench | HuggingFace + GitHub | 1055+ problems (v6) | Code generation | `load_dataset("livecodebench/code_generation_lite")` | Ready to download |
| LiveBench | HuggingFace + GitHub | Growing (monthly) | Multi-domain reasoning | `load_dataset("livebench/coding")` | Ready to download |
| OpenOOD | GitHub (manual setup) | Multiple benchmarks | Out-of-distribution detection | See OpenOOD repo instructions | Requires setup |

See `datasets/README.md` for detailed download instructions and usage examples.

### Dataset Categories by Contamination Resistance

**Highest Resistance (Temporal)**:
- LiveCodeBench: Continuous updates, release date filtering
- LiveBench: Monthly updates from recent sources

**High Resistance (Rewritten)**:
- MMLU-CF: Systematic contamination-free rewriting
- MMLU-Pro: Increased difficulty (10 options vs. 4)

**High Resistance (Synthetic)**:
- GSM-Symbolic: Template-based infinite variants

**Complementary (OOD)**:
- OpenOOD: Distribution shift testing

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location | Stars/Activity |
|------|-----|---------|----------|----------------|
| MMLU-CF | github.com/microsoft/MMLU-CF | Official evaluation code for MMLU-CF | code/MMLU-CF/ | Microsoft Research |
| LiveCodeBench | github.com/LiveCodeBench/LiveCodeBench | Holistic contamination-free code evaluation | code/LiveCodeBench/ | Active development |
| LiveBench | github.com/LiveBench/LiveBench | Challenging contamination-free LLM benchmark | code/LiveBench/ | Active development |
| OpenOOD | github.com/Jingkang50/OpenOOD | OOD detection benchmark suite | code/OpenOOD/ | 1.8K+ stars |
| awesome-data-contamination | github.com/lyy1994/awesome-data-contamination | Curated paper list on contamination | code/awesome-data-contamination/ | Regularly updated |

See `code/README.md` for detailed descriptions, key files, and usage instructions.

### Repository Dependencies

**General Requirements**:
- Python 3.8+
- PyTorch 1.10+
- Transformers library
- HuggingFace datasets library

**Specific Requirements**:
- LiveCodeBench: Execution environments for multiple languages
- OpenOOD: Computer vision libraries (torchvision, timm)
- MMLU-CF: OpenAI/Anthropic API keys for evaluation

## Resource Gathering Notes

### Search Strategy

**Phase 1: Literature Search**
- Searched arXiv, Semantic Scholar, Papers with Code
- Keywords: "finetuning proof datasets", "benchmark saturation", "data contamination", "train-test overlap", "dynamic benchmarks", "robust evaluation"
- Timeframe focus: 2023-2025 (recent developments)
- Result: 9 highly relevant papers covering contamination detection, benchmark design, and evaluation methodologies

**Phase 2: Dataset Search**
- Searched HuggingFace datasets, GitHub, benchmark websites
- Focused on: Contamination-resistant, temporal, rewritten, and OOD datasets
- Prioritized: Established benchmarks with code availability
- Result: 6 primary datasets spanning multiple contamination-resistance strategies

**Phase 3: Code Repository Search**
- Searched GitHub for official implementations
- Verified: Repository activity, documentation quality, ease of use
- Prioritized: Repositories with active maintenance and clear documentation
- Result: 5 key repositories plus 1 curated resource list

### Selection Criteria

**Papers Selected Based On**:
1. Direct relevance to finetuning-proof dataset research
2. Recent publication (2023-2025 preferred)
3. Strong methodology and empirical results
4. Availability of associated datasets/code
5. Publication venue quality (ICLR, NeurIPS, ACL)

**Datasets Selected Based On**:
1. Demonstrated contamination resistance
2. Active use in recent research
3. Availability and ease of access (HuggingFace preferred)
4. Multiple evaluation domains covered
5. Documented download and usage procedures

**Code Selected Based On**:
1. Official implementations from paper authors
2. Active maintenance and documentation
3. Community adoption (GitHub stars, citations)
4. Clear integration path for experiments
5. Permissive licensing

### Challenges Encountered

**Challenge 1: Temporal Benchmark Dynamics**
- **Issue**: Temporal benchmarks continuously update, making exact replication difficult
- **Solution**: Document specific versions (e.g., LiveCodeBench v6) and release date ranges
- **Impact**: Minor - versions are well-documented

**Challenge 2: Dataset Size**
- **Issue**: Some datasets (OpenOOD with ImageNet) are very large (100GB+)
- **Solution**: Documented manual download procedures; created .gitignore to exclude from repo
- **Impact**: Moderate - requires separate download step, but well-documented

**Challenge 3: Contamination Detection Complexity**
- **Issue**: No single method detects all contamination types
- **Solution**: Recommended multi-pronged approach using temporal + rewritten + symbolic datasets
- **Impact**: Minor - complexity is inherent to the research question

**Challenge 4: Rapidly Evolving Field**
- **Issue**: New papers and benchmarks released frequently (February 2025 survey!)
- **Solution**: Focused on established benchmarks with strong empirical validation
- **Impact**: Positive - field is active and well-resourced

### Gaps and Workarounds

**Gap 1: Limited Domain Coverage**
- **Description**: Most contamination-resistant benchmarks focus on code, math, and general language understanding
- **Missing**: Medicine, law, science-specific contamination-resistant benchmarks
- **Workaround**: Use general benchmarks (MMLU-CF) which include these subjects
- **Future Work**: Develop domain-specific temporal benchmarks

**Gap 2: Multimodal Contamination**
- **Description**: Limited research on vision-language, audio-text contamination
- **Missing**: Robust multimodal benchmarks
- **Workaround**: Focus on text-only evaluation for initial research
- **Future Work**: Extend findings to multimodal settings

**Gap 3: Small Model Studies**
- **Description**: Most research focuses on large frontier models (GPT-4, Claude)
- **Missing**: Systematic contamination studies across model scales
- **Workaround**: One paper (2511.06763) covers 270M-4B models; provides baseline
- **Future Work**: Extend contamination testing to full model scale spectrum

**Gap 4: Standardized Reporting**
- **Description**: No universal standard for reporting train-test overlap
- **Missing**: Standardized contamination metrics and reporting formats
- **Workaround**: Use multiple detection methods (performance gaps, variance, temporal)
- **Future Work**: Propose standardized reporting framework

## Recommendations for Experiment Design

Based on the gathered resources, we recommend the following experimental approach:

### 1. Primary Datasets (Priority 1)

**LiveCodeBench (v6)**:
- **Why**: Highest contamination resistance via temporal filtering
- **Use**: Evaluate on problems released after model training cutoff
- **Metric**: Pass@k, accuracy
- **Expected Outcome**: Minimal fine-tuning improvement on post-cutoff problems indicates true finetuning-proof properties

**MMLU-CF**:
- **Why**: Proven contamination-free with documented performance gaps
- **Use**: Compare performance on MMLU vs. MMLU-CF
- **Metric**: Accuracy, performance gap
- **Expected Outcome**: Large gaps indicate contamination in original MMLU

**GSM-Symbolic**:
- **Why**: Detects memorization vs. reasoning through variance
- **Use**: Test on multiple symbolic variants (main, P1, P2)
- **Metric**: Mean accuracy, variance across variants
- **Expected Outcome**: High variance indicates memorization; true reasoning shows consistent performance

### 2. Secondary Datasets (Priority 2)

**MMLU-Pro**: Additional language understanding evaluation with harder questions

**LiveBench**: Broader domain coverage beyond code

### 3. Complementary Datasets (Priority 3)

**OpenOOD**: Test if fine-tuning improves robustness vs. just memorization

### 4. Baseline Methods

**Models to Evaluate**:
1. GPT-4 / GPT-4o (if API access available)
2. Claude 3.5 Sonnet (if API access available)
3. Llama 3 (8B, 70B) - open-source baseline
4. Mistral 7B - smaller open-source baseline

**Evaluation Protocol**:
1. Baseline evaluation on all datasets (pre-fine-tuning)
2. Fine-tune on dataset training splits
3. Re-evaluate on test sets
4. Calculate:
   - Performance improvement
   - Performance gaps (original vs. contamination-free)
   - Variance across symbolic variants
   - Temporal performance trends

### 5. Evaluation Metrics

**Primary**:
- **Accuracy** (or task-specific: Pass@k for code, exact match for math)
- **Performance Gap**: Δ = Performance(original) - Performance(contamination-free)
- **Variance**: σ across symbolic variants or temporal windows

**Secondary**:
- **Perturbation Sensitivity**: Performance with irrelevant information added
- **Temporal Degradation**: Performance trend vs. release date
- **OOD Robustness**: AUROC, AUPR, FPR@95

### 6. Methodological Considerations

**Critical Requirements**:
1. **Document Training Cutoffs**: Essential for temporal evaluation
2. **Use Multiple Variants**: Test on multiple symbolic instantiations
3. **Report All Results**: Include failures and variance, not just best performance
4. **Statistical Testing**: Multiple seeds, confidence intervals
5. **Transparency**: Document any potential train-test overlap

**Experiment Design**:
```
For each model M:
    For each dataset D:
        1. Baseline: Evaluate M on D_test
        2. Fine-tune: Train M on D_train
        3. Re-evaluate: Evaluate M_finetuned on D_test
        4. Compare:
            - If D has contamination-free variant D_cf:
                Calculate gap = Performance(D) - Performance(D_cf)
            - If D has symbolic variants:
                Calculate variance across variants
            - If D has temporal annotations:
                Evaluate only on post-cutoff data
        5. Classify:
            - Small improvement (<5%) → High finetuning-proof
            - Medium improvement (5-15%) → Moderate finetuning-proof
            - Large improvement (>15%) → Low finetuning-proof (likely memorization)
```

**Success Criteria**:
A dataset is "finetuning-proof" if:
- Fine-tuning improvement < 5% absolute
- Low variance across symbolic variants (< 2% std dev)
- Consistent performance on temporal splits
- Similar performance on contamination-free variants

### 7. Code Integration Path

**Recommended Workflow**:
1. Start with **LiveCodeBench** (well-documented, clear API)
2. Add **MMLU-CF** (straightforward evaluation)
3. Integrate **GSM-Symbolic** (multiple variants)
4. Optionally add OpenOOD for robustness testing

**Implementation Steps**:
```bash
# 1. Setup environment
pip install datasets transformers torch

# 2. Download datasets
python scripts/download_datasets.py  # Create this script

# 3. Run baseline evaluation
python experiments/evaluate_baseline.py --model llama3-8b --dataset livecodebench

# 4. Fine-tune models
python experiments/finetune.py --model llama3-8b --dataset livecodebench-train

# 5. Re-evaluate and compare
python experiments/evaluate_finetuned.py --model llama3-8b --dataset livecodebench

# 6. Analyze results
python analysis/compute_finetuning_proof_scores.py --results results/
```

## Summary Statistics

### Resource Coverage

- **Papers**: 9 papers covering contamination detection, benchmark design, evaluation methodologies
- **Datasets**: 6 primary datasets + multiple OOD variants
- **Code Repositories**: 5 repositories + 1 resource list
- **Time Period**: 2023-2025 (recent, cutting-edge research)
- **Domains Covered**: Code, math, language understanding, vision, multi-domain reasoning

### Quality Indicators

- **Publication Venues**: ICLR 2025, ACL 2025, NeurIPS 2022/2024, top-tier arXiv papers
- **Industry Backing**: Microsoft Research, Apple, Google, academic institutions
- **Community Adoption**: Active GitHub repositories, HuggingFace datasets
- **Documentation Quality**: High - all resources well-documented with examples

### Completeness Assessment

**Strong Coverage** ✓:
- Data contamination detection methods
- Contamination-resistant benchmark designs
- Temporal evaluation approaches
- Mathematical reasoning evaluation
- Code generation evaluation

**Adequate Coverage** ~:
- Out-of-distribution robustness testing
- Multi-task language understanding
- Dynamic benchmark maintenance

**Limited Coverage** ○:
- Domain-specific benchmarks (medicine, law, science)
- Multimodal contamination
- Small model contamination studies
- Cross-lingual contamination

### Readiness for Experimentation

**Ready** ✓:
- All datasets accessible via HuggingFace or documented downloads
- Evaluation code available in repositories
- Clear baseline models and metrics identified
- Methodological framework established

**Requires Setup** ⚙:
- OpenOOD datasets (large downloads)
- Model fine-tuning infrastructure
- API access for frontier models (GPT-4, Claude)

**Future Work** ⏭:
- Standardized contamination reporting
- Automated benchmark generation
- Cross-modal evaluation

## Conclusion

This resource gathering phase has successfully identified and documented:

1. **9 high-quality research papers** establishing the state of contamination detection and finetuning-proof benchmark design
2. **6 robust datasets** with varying contamination-resistance strategies (temporal, rewritten, symbolic, OOD)
3. **5 code repositories** providing implementation and evaluation infrastructure
4. **Clear experimental roadmap** for testing finetuning-proof properties

**Key Insight**: No dataset is perfectly finetuning-proof, but **temporal benchmarks** (LiveCodeBench, LiveBench) show highest resistance through continuous updates and release date filtering.

**Research Answer**: "Are there any finetuning-proof datasets currently?"
- **Short Answer**: Yes, several datasets show strong resistance, particularly temporal benchmarks
- **Nuanced Answer**: "Finetuning-proof" is a spectrum, not binary. Temporal benchmarks are most robust, followed by contamination-free rewrites and symbolic variants. All require maintenance and multi-pronged evaluation.

**Next Phase**: The experiment runner can now use these resources to:
1. Download datasets systematically
2. Implement evaluation pipeline using provided code
3. Run fine-tuning experiments
4. Quantify "finetuning-proof" properties across datasets
5. Validate findings from literature review empirically

All resources are documented, accessible, and ready for automated experimentation.

---

**Resource Gathering Completed**: December 1, 2025
**Total Time**: ~2-3 hours (within budget)
**Status**: ✓ Complete - Ready for experiment runner phase
