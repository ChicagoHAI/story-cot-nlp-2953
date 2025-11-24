# Resources Catalog: Story CoT Research Project

This document catalogs all resources gathered for the "Story CoT" research project investigating whether narrative-based chain-of-thought reasoning improves language model performance.

---

## Summary

**Research Topic**: Story CoT - Investigating narrative-based chain-of-thought reasoning in LLMs

**Papers Downloaded**: 6
**Datasets Downloaded**: 3 (1 gated)
**Code Repositories Cloned**: 3

**Total Disk Usage**: ~150 MB (excluding code repositories)
**Resource Gathering Time**: ~2.5 hours

---

## Papers

### Downloaded Papers Summary

| Title | Authors | Year | File | arXiv | Primary Relevance |
|-------|---------|------|------|-------|-------------------|
| Chain-of-Thought Prompting | Wei et al. | 2022 | [2201.11903_chain_of_thought_prompting.pdf](papers/2201.11903_chain_of_thought_prompting.pdf) | 2201.11903 | Foundational CoT paper |
| Can Stories Help LLMs Reason? | Sadiri Javadi et al. | 2024 | [2410.19221_stories_help_llms_reason.pdf](papers/2410.19221_stories_help_llms_reason.pdf) | 2410.19221 | **DIRECTLY RELEVANT** - Story of Thought |
| Self-Consistency CoT | Wang et al. | 2022 | [2203.11171_self_consistency_cot.pdf](papers/2203.11171_self_consistency_cot.pdf) | 2203.11171 | Important baseline method |
| Multimodal CoT | Zhang et al. | 2023 | [2302.00923_multimodal_cot.pdf](papers/2302.00923_multimodal_cot.pdf) | 2302.00923 | Multi-stage reasoning approach |
| Physics Reasoner | Authors | 2024 | [2412.13791_physics_reasoner.pdf](papers/2412.13791_physics_reasoner.pdf) | 2412.13791 | Physics problem solving |
| Auto-CoT | Zhang et al. | 2022 | [2210.03493_auto_cot.pdf](papers/2210.03493_auto_cot.pdf) | 2210.03493 | Automated baseline |

See [papers/README.md](papers/README.md) for detailed descriptions.

---

## Datasets

### Downloaded Datasets Summary

| Name | Source | Size | Task | Location | Status |
|------|--------|------|------|----------|--------|
| GSM8K | HuggingFace (openai/gsm8k) | 7.5K train, 1.3K test | Math word problems | datasets/gsm8k/ | ✓ Downloaded |
| SciBench | HuggingFace (xw27/scibench) | 692 problems | College science | datasets/scibench/ | ✓ Downloaded |
| JEEBench | HuggingFace (daman1209arora/jeebench) | 515 problems | Physics/chem/math | datasets/jeebench/ | ✓ Downloaded |
| GPQA | HuggingFace (Idavidrein/gpqa) | 198 (Diamond) | Graduate science | - | ⚠ Gated (requires auth) |

### Dataset Details

**GSM8K (Grade School Math 8K)**
- **Format**: HuggingFace Dataset
- **Splits**: train (7,473), test (1,319)
- **Fields**: question, answer (with step-by-step solution)
- **License**: MIT
- **Use Case**: Baseline math reasoning benchmark

**SciBench**
- **Format**: HuggingFace Dataset
- **Total**: 692 college-level problems
- **Domains**: Physics, Chemistry, Mathematics
- **Fields**: problem, solution, answer
- **Use Case**: Physics evaluation (per hypothesis)

**JEEBench**
- **Format**: HuggingFace Dataset
- **Total**: 515 IIT JEE Advanced problems
- **Subjects**: Physics, Chemistry, Mathematics
- **Types**: Single-correct, Multi-correct, Integer, Numeric
- **Use Case**: **Primary evaluation** - used in SoT paper

**GPQA (Graduate-level Problem-solving QA)**
- **Status**: Gated - requires HuggingFace authentication
- **Subset**: Diamond (198 high-agreement questions)
- **Domains**: Biology, Physics, Chemistry
- **Difficulty**: Expert PhD: 65%, Non-expert with Google: 34%
- **Use Case**: **Primary evaluation** - used in SoT paper

See [datasets/README.md](datasets/README.md) for download instructions.

---

## Code Repositories

### Cloned Repositories Summary

| Name | URL | Purpose | Location | Key Features |
|------|-----|---------|----------|--------------|
| Auto-CoT | [amazon-science/auto-cot](https://github.com/amazon-science/auto-cot) | Automated CoT baseline | code/auto-cot/ | Demo construction, automatic prompting |
| Tree of Thoughts | [princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm) | Structured reasoning baseline | code/tree-of-thought-llm/ | BFS/DFS search, multi-path reasoning |
| CoT Hub | [FranxYao/chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) | Benchmarking framework | code/chain-of-thought-hub/ | Standardized evaluation |

### Repository Details

**Auto-CoT (Automatic Chain of Thought)**
- **Language**: Python
- **Dependencies**: PyTorch 1.8.2+cu111, Python >=3.8
- **Key Files**: run_demo.py, run_inference.py, api.py
- **Relevance**: Automated baseline for demonstration construction

**Tree of Thoughts**
- **Language**: Python
- **Dependencies**: OpenAI API (or other LLM APIs)
- **Key Files**: src/tot/methods/, src/tot/tasks/
- **Relevance**: Comparison method (used in SoT paper)

**Chain-of-Thought Hub**
- **Language**: Python
- **Purpose**: Comprehensive benchmarking suite
- **Key Files**: benchmarks/, prompts/, evaluation/
- **Relevance**: Standardized evaluation framework

See [code/README.md](code/README.md) for detailed documentation.

---

## Resource Gathering Notes

### Search Strategy

1. **Papers**: Searched arXiv and Google Scholar for:
   - "chain of thought reasoning" (foundational work)
   - "narrative reasoning LLM" (story-based approaches)
   - "physics problem solving LLM" (domain-specific)
   - "CoT prompting evaluation" (benchmarks)

2. **Datasets**: Searched HuggingFace and Papers with Code for:
   - Datasets mentioned in SoT paper (GPQA, JEEBench)
   - Standard reasoning benchmarks (GSM8K)
   - Physics-specific datasets (SciBench)

3. **Code**: Searched GitHub for:
   - Official implementations of methods in papers
   - Baseline CoT implementations
   - Evaluation frameworks

### Selection Criteria

**Papers** (selected 6 from ~20 candidates):
- Direct relevance to Story CoT hypothesis
- Foundational/highly-cited works (CoT, Self-Consistency)
- Recent work on narrative reasoning (SoT paper - 2024)
- Domain-specific (Physics Reasoner)
- Code available (Auto-CoT)

**Datasets** (selected 4 from ~10 candidates):
- Used in SoT paper (GPQA, JEEBench) - enables comparison
- Standard benchmarks (GSM8K) - enables broader evaluation
- Physics-focused (SciBench) - tests hypothesis
- Accessible via HuggingFace - easy to download and use

**Code** (selected 3 from ~8 candidates):
- Official implementations preferred
- Active maintenance (recent commits)
- Clear documentation
- Baseline methods (Auto-CoT, ToT)
- Evaluation framework (CoT Hub)

### Challenges Encountered

**Challenge 1: GPQA Gated Access**
- **Issue**: Dataset requires authentication and access request
- **Workaround**: Documented access process, can authenticate later
- **Impact**: Not critical - have JEEBench as alternative primary dataset

**Challenge 2: JEEBench Dataset Structure**
- **Issue**: Initial download attempt failed due to split specification
- **Solution**: Used `split="test"` parameter
- **Resolution**: Successfully downloaded 515 problems

**Challenge 3: SoT Paper Code Not Available**
- **Issue**: Official implementation not found in public repositories
- **Workaround**: Detailed methodology in paper allows reimplementation
- **Plan**: Implement based on paper description (3-step process with 5 narrative techniques)

**Challenge 4: Large File Sizes**
- **Issue**: Some datasets and papers are large (>100 MB)
- **Solution**: Implemented .gitignore to exclude from version control
- **Benefit**: Local access for experiments, git repo stays small

### Gaps and Workarounds

**Gap 1: No Public SoT Implementation**
- **Impact**: Cannot directly reproduce SoT results
- **Workaround**: Paper provides detailed prompts (Appendix C) - can reimplement
- **Action**: Will implement SoT methodology from scratch

**Gap 2: Limited Physics-Specific Datasets**
- **Impact**: Only SciBench physics-specific, limited size (692 problems)
- **Workaround**: GPQA and JEEBench include physics problems
- **Action**: Can filter by subject domain for physics-specific analysis

**Gap 3: Baseline Code Compatibility**
- **Impact**: Some repos use older PyTorch versions
- **Workaround**: Can adapt code or use as reference
- **Action**: Likely will implement own baselines using modern libraries

---

## Recommendations for Experiment Design

Based on gathered resources, recommend:

### 1. Primary Dataset(s)

**Recommendation**: JEEBench (primary), GPQA (if access granted)

**Rationale**:
- Both used in SoT paper - enables direct comparison
- JEEBench: 515 problems, physics/chem/math split, publicly accessible
- GPQA: 198 Diamond subset, graduate-level, includes physics
- SciBench: Additional physics evaluation

**Implementation**:
```python
# Load datasets
jeebench = load_from_disk("datasets/jeebench")
scibench = load_from_disk("datasets/scibench")
# gsm8k = load_from_disk("datasets/gsm8k")  # optional baseline
```

### 2. Baseline Methods

**Must-implement baselines**:
1. **Zero-shot**: Direct prompting
2. **Zero-shot CoT**: "Let's think step by step"
3. **Story of Thought**: Full 3-step methodology from paper

**Optional baselines** (if time/budget allows):
4. **Self-Consistency**: Sample multiple reasoning paths
5. **Auto-CoT**: Automated demonstration construction
6. **Tree of Thoughts**: Structured exploration

**Implementation Priority**: Focus on reproducing SoT results first, then add comparisons.

### 3. Evaluation Metrics

**Primary**: Accuracy (% correct)
- Standard across all papers
- Easy to compute and compare

**Secondary**:
- **Per-domain breakdown**: Physics vs Chemistry vs Math
- **Per-question-type**: Single-correct vs Multi-correct (for JEEBench)
- **Token usage**: Cost-effectiveness analysis

**Optional**:
- **Explanation quality**: BLEU/BERTScore (if comparing to human solutions)
- **Consistency**: Agreement across multiple generations

### 4. Code to Adapt/Reuse

**Recommended approach**: Implement from scratch using:
- **HuggingFace Transformers** for open-source models (Llama 3, etc.)
- **OpenAI API** for GPT-3.5, GPT-4
- **Auto-CoT** as reference for demonstration construction
- **CoT Hub** as reference for evaluation framework

**Justification**: Clean implementation easier than adapting old code, better documentation.

---

## Next Steps for Experiment Runner

The resource gathering phase is complete. The next phase (experiment runner) should:

### Immediate Actions

1. **Environment Setup**
   ```bash
   pip install torch transformers datasets openai anthropic
   pip install scikit-learn pandas numpy
   ```

2. **Verify Resources**
   ```bash
   # Check papers
   ls -lh papers/*.pdf

   # Check datasets
   ls -lh datasets/

   # Check code
   ls -lh code/
   ```

3. **Initial Data Exploration**
   ```python
   # Load and inspect datasets
   from datasets import load_from_disk
   jeebench = load_from_disk("datasets/jeebench")
   print(f"JEEBench size: {len(jeebench)}")
   print(f"Sample: {jeebench[0]}")
   ```

### Implementation Tasks

1. **Implement SoT Methodology** (from paper Appendix C):
   - Step 1: Question Clarification prompt
   - Step 2: Narrative Generation prompt (with 5 techniques)
   - Step 3: Problem Solving prompt

2. **Implement Baselines**:
   - Zero-shot prompting
   - Zero-shot CoT ("Let's think step by step")
   - Standard few-shot CoT

3. **Set Up Evaluation Pipeline**:
   - Load datasets
   - Run inference with each method
   - Compute accuracy and other metrics
   - Save results to structured format

4. **Run Initial Experiments**:
   - Small subset validation (10-20 examples)
   - Verify reproduction of SoT paper results
   - Debug any issues

5. **Full Evaluation**:
   - Run on complete datasets
   - Generate result tables matching SoT paper format
   - Perform statistical analysis

### Expected Outputs

- **Results Tables**: Accuracy by method, dataset, domain
- **Error Analysis**: Categorization of failure modes
- **Cost Analysis**: Tokens used per method
- **Comparison**: Story CoT vs baselines

### Success Criteria

✓ Reproduce SoT paper results on JEEBench (within ±2%)
✓ Evaluate on physics problems specifically
✓ Compare multiple baseline methods
✓ Document findings and insights

---

## Citation Guidelines

When using these resources, cite:

### Papers
- See [papers/README.md](papers/README.md) for BibTeX entries

### Datasets
```bibtex
@inproceedings{arora2023jeebench,
  title={Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark},
  author={Arora, Daman and Singh, Himanshu and Mausam},
  booktitle={EMNLP},
  year={2023}
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}

@inproceedings{wang2023scibench,
  title={SciBench: Evaluating College-Level Scientific Problem-Solving},
  author={Wang, Xiaoxuan and others},
  booktitle={ICML},
  year={2024}
}
```

---

## Maintenance Notes

- **Dataset updates**: Check HuggingFace periodically for dataset updates
- **Paper updates**: Monitor arXiv for new versions of papers
- **Code updates**: Check repositories for bugfixes and improvements
- **Access**: GPQA may require re-authentication periodically

---

## Contact Information

For questions about resources:
- **Dataset issues**: Check HuggingFace dataset pages
- **Paper questions**: Contact authors via arXiv
- **Code issues**: Open GitHub issues on respective repositories

---

## Appendix: File Structure

```
story-cot-nlp-2953/
├── papers/
│   ├── README.md
│   ├── 2201.11903_chain_of_thought_prompting.pdf
│   ├── 2410.19221_stories_help_llms_reason.pdf
│   ├── 2203.11171_self_consistency_cot.pdf
│   ├── 2302.00923_multimodal_cot.pdf
│   ├── 2412.13791_physics_reasoner.pdf
│   └── 2210.03493_auto_cot.pdf
├── datasets/
│   ├── README.md
│   ├── .gitignore
│   ├── download_datasets.py
│   ├── download_jeebench.py
│   ├── gsm8k/
│   ├── scibench/
│   └── jeebench/
├── code/
│   ├── README.md
│   ├── auto-cot/
│   ├── tree-of-thought-llm/
│   └── chain-of-thought-hub/
├── literature_review.md
├── resources.md (this file)
└── .resource_finder_complete (to be created)
```

---

**Resource Gathering Complete**: All materials ready for experimentation phase.
