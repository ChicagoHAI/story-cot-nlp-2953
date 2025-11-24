# Story CoT Research Project - Resource Gathering Summary

## Status: ✅ COMPLETE

The resource finding phase has been successfully completed. All necessary materials have been gathered and documented for the experiment runner to begin implementation.

## What Was Accomplished

### 1. Literature Review ✅
- **6 key papers identified and reviewed**
- Most importantly: Found the **Story of Thought (SoT) paper** which directly addresses our hypothesis
- Created comprehensive literature review with methodology synthesis
- Documented baselines, metrics, and experimental approaches

### 2. Datasets ✅
- **3 datasets downloaded** and ready for use:
  - GSM8K (7,473 train, 1,319 test) - math reasoning baseline
  - JEEBench (515 problems) - **PRIMARY DATASET** for evaluation
  - SciBench (692 problems) - physics-focused evaluation
- **1 dataset documented** (GPQA - requires authentication)
- All datasets stored locally in `datasets/` directory
- Created comprehensive documentation with download/usage instructions

### 3. Code Repositories ✅
- **3 baseline implementations cloned**:
  - Auto-CoT - automated chain-of-thought
  - Tree of Thoughts - structured reasoning
  - Chain-of-Thought Hub - evaluation framework
- All code in `code/` directory with documentation

### 4. Documentation ✅
Created 5 comprehensive markdown files:
- `literature_review.md` - Full literature analysis and synthesis
- `resources.md` - Complete resource catalog
- `papers/README.md` - Paper descriptions
- `datasets/README.md` - Dataset documentation with examples
- `code/README.md` - Repository documentation

## Key Finding: Story of Thought (SoT) Paper

**The most important discovery**: A 2024 paper by Sadiri Javadi et al. (arXiv:2410.19221) titled "Can Stories Help LLMs Reason?" introduces the exact approach our hypothesis proposes!

**SoT Methodology** (3 steps):
1. Question Clarification - Break down problem into components
2. Narrative Generation - Use 5 techniques (Progressive Disclosure, Branching, Analogy, Analogical Reasoning, Metaphor)
3. Problem Solving - Solve using the generated narrative

**Results**:
- Llama 3 70B: 51.01% on GPQA (vs 39.5% zero-shot)
- GPT-4: 48.98% on GPQA (vs 34.7% zero-shot) - **41% relative improvement**
- Outperformed all baselines including Tree of Thoughts and standard CoT

This provides both validation of our hypothesis AND a concrete methodology to implement/build upon.

## Resources Ready for Experimentation

### Datasets (Locally Available)
```
datasets/
├── gsm8k/          # 8,792 total examples
├── jeebench/       # 515 problems (PRIMARY)
└── scibench/       # 692 problems
```

### Code References
```
code/
├── auto-cot/                # Auto-CoT implementation
├── tree-of-thought-llm/     # ToT implementation  
└── chain-of-thought-hub/    # Evaluation framework
```

### Documentation
```
./
├── literature_review.md     # Comprehensive review
├── resources.md            # Resource catalog
├── datasets/README.md      # Dataset guide
├── code/README.md          # Code documentation
└── .resource_finder_complete # Completion marker
```

## Recommendations for Experiment Runner

### Primary Evaluation Dataset
**JEEBench** (515 problems)
- Publicly accessible
- Used in SoT paper (enables direct comparison)
- Physics, Chemistry, Mathematics problems
- Multiple question types

### Baseline Methods to Implement
1. **Zero-shot** - Direct prompting (simplest baseline)
2. **Zero-shot CoT** - "Let's think step by step"
3. **Story of Thought** - 3-step narrative approach from SoT paper

### Success Criteria
- Reproduce SoT paper results on JEEBench (within ±2%)
- Evaluate specifically on physics problems
- Test if narrative helps more on physics vs. other domains
- Document which narrative elements are most effective

## Next Steps for Experiment Runner

1. **Environment Setup**
   ```bash
   pip install transformers datasets openai anthropic torch
   ```

2. **Verify Resources**
   ```python
   from datasets import load_from_disk
   jeebench = load_from_disk("datasets/jeebench")
   print(f"Loaded {len(jeebench)} problems")
   ```

3. **Implement SoT Methodology**
   - Follow prompts from SoT paper Appendix C
   - 3-step process with 5 narrative techniques

4. **Run Initial Validation**
   - Test on small subset (10-20 problems)
   - Verify correctness of implementation

5. **Full Evaluation**
   - Run on complete JEEBench dataset
   - Compare with baselines
   - Analyze results by domain (physics vs. chemistry vs. math)

## Files Inventory

| File/Directory | Purpose | Status |
|----------------|---------|--------|
| datasets/gsm8k/ | Math word problems | ✅ Downloaded |
| datasets/jeebench/ | Primary evaluation set | ✅ Downloaded |
| datasets/scibench/ | Physics problems | ✅ Downloaded |
| code/auto-cot/ | Auto-CoT implementation | ✅ Cloned |
| code/tree-of-thought-llm/ | ToT implementation | ✅ Cloned |
| code/chain-of-thought-hub/ | Evaluation framework | ✅ Cloned |
| literature_review.md | Literature synthesis | ✅ Created |
| resources.md | Resource catalog | ✅ Created |
| datasets/README.md | Dataset documentation | ✅ Created |
| code/README.md | Code documentation | ✅ Created |
| .resource_finder_complete | Completion marker | ✅ Created |

## Time and Resource Budget

- **Time Spent**: ~2.5 hours
- **Disk Usage**: ~150 MB (datasets + code)
- **Budget Used**: Minimal (web searches, downloads only)
- **Budget Remaining**: 100 (full budget available for experiments)

## Critical Insights for Experimentation

1. **Physics problems are particularly challenging** - Standard CoT achieves only 6.8% on SciBench physics (GPT-3.5-turbo)

2. **Narrative approaches show strong promise** - SoT consistently outperformed all baselines across multiple models

3. **Combination of techniques matters** - Using all 5 narrative techniques together more effective than any single technique

4. **Domain-specific patterns** - Biology showed most improvement with SoT, followed by chemistry

5. **Model size matters** - Larger models benefited more from narrative generation

## Questions for Further Investigation

Based on literature review, these questions remain:

1. Does "story of past experience" framing improve over general narrative?
2. Which narrative elements are most effective for physics specifically?
3. Can we identify patterns in when narrative helps vs. doesn't?
4. How does token cost compare to accuracy improvement?

---

## Conclusion

✅ **Resource gathering phase COMPLETE and SUCCESSFUL**

All necessary materials are in place for the experiment runner to:
- Implement Story of Thought methodology
- Evaluate on physics/science reasoning benchmarks
- Compare against standard CoT baselines
- Test the core hypothesis about narrative-based reasoning

The discovery of the SoT paper provides both validation of the research direction and a concrete methodology to build upon. The experiment runner can now proceed with confidence that the approach is well-grounded in recent research.

**Ready for experimentation!** 🚀
