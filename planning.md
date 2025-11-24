# Research Plan: Story CoT - Narrative-Based Chain-of-Thought Reasoning

## Research Question

**Primary Question**: Does training or prompting language models to use narrative-based chain-of-thought (CoT) reasoning—similar to how humans recount memories or tell stories about situations—result in improved performance compared to standard CoT approaches, particularly on physics problem modeling tasks?

**Sub-questions**:
1. Can current LLMs produce story-like CoTs when prompted appropriately?
2. Does story-based CoT improve performance on physics problems specifically?
3. What narrative elements are most effective for improving reasoning?
4. Do different models benefit differently from narrative-based prompting?

## Background and Motivation

### Why This Matters

Current CoT methods treat reasoning as a logical chain of steps, but human reasoning often involves narrative structures—recounting past experiences, imagining scenarios, and connecting ideas through stories. The recent Story of Thought (SoT) paper demonstrates that integrating narrative structures significantly improves LLM performance on graduate-level science problems (41% relative improvement on GPQA).

This research is important because:
- **Cognitive Science Foundation**: Humans naturally think in narratives and stories
- **Gap in Current Methods**: Standard CoT doesn't leverage narrative structures
- **Physics Domain Challenge**: Physics problems are particularly hard for LLMs (6.8% accuracy with standard CoT on SciBench)
- **Novel Approach**: Story-based reasoning is underexplored in LLM research

### What Gap This Fills

While SoT introduces narrative-based prompting, several questions remain:
1. **Domain specificity**: Does narrative help more on physics vs. other subjects?
2. **Narrative style variations**: Are there better narrative framings (e.g., "memory recall" vs. "story explanation")?
3. **Simplified approaches**: Can we achieve similar benefits with simpler single-step prompts?
4. **Model comparison**: How do different models respond to narrative prompting?

## Hypothesis Decomposition

### Primary Hypothesis
**H1**: LLMs prompted with story-based CoT will achieve higher accuracy on physics problems than standard CoT methods.

### Sub-Hypotheses
**H1a**: Story-based CoT will show greater improvement on physics problems compared to chemistry or math problems (domain specificity).

**H1b**: Different narrative framings (e.g., "memory recall" vs. "analogical story") will have varying effectiveness.

**H1c**: Larger models will benefit more from narrative prompting than smaller models.

**H1d**: Story-based CoT will maintain or improve performance while being more human-interpretable than standard CoT.

### Success Criteria
- **Minimum**: Reproduce SoT paper results on JEEBench (within ±3%)
- **Good**: Show statistically significant improvement on physics subset
- **Excellent**: Identify specific narrative techniques that drive performance gains

## Proposed Methodology

### High-Level Strategy

We will conduct a **comparative evaluation study** using multiple prompting methods on science problem datasets, with particular focus on physics problems. This is a **prompting-based** approach (not training) due to time and budget constraints.

**Rationale**:
- Prompting allows rapid experimentation with different narrative styles
- Can use state-of-the-art models (GPT-4, GPT-4.1) without training costs
- Enables direct comparison with SoT paper results
- Focuses resources on testing the hypothesis rather than infrastructure

### Experimental Design

#### Phase 1: Baseline Implementation (15-20 min)
1. Implement zero-shot prompting
2. Implement zero-shot CoT ("Let's think step by step")
3. Implement few-shot CoT with examples
4. Validate on small subset (10 examples)

**Rationale**: Establish performance floor and verify evaluation pipeline works correctly.

#### Phase 2: Story of Thought Implementation (30-40 min)
1. Implement SoT 3-step process:
   - Step 1: Question Clarification
   - Step 2: Narrative Generation (5 techniques)
   - Step 3: Problem Solving
2. Test on small subset
3. Compare with baseline results

**Rationale**: Reproduce SoT methodology to validate we can achieve similar performance.

#### Phase 3: Novel Narrative Variations (20-30 min)
1. Design "Memory Recall" framing: "Recall a similar problem you've seen..."
2. Design "Story Explanation" framing: "Tell a story that explains this situation..."
3. Design "Single-Step Narrative" framing: Combine all steps into one prompt
4. Test variations

**Rationale**: Test whether simpler or different narrative framings can match or exceed SoT performance.

#### Phase 4: Full Evaluation (30-40 min)
1. Run all methods on full datasets
2. Collect results systematically
3. Generate performance tables
4. Statistical analysis

**Rationale**: Comprehensive evaluation to test hypotheses.

### Datasets

#### Primary Dataset: JEEBench (515 problems)
- **Why**: Used in SoT paper, enables direct comparison
- **Coverage**: Physics (171), Chemistry (172), Mathematics (172)
- **Difficulty**: IIT JEE Advanced level (extremely challenging)
- **Format**: Single-correct, multi-correct, integer, numeric answer types
- **Advantage**: Can analyze performance by subject domain

#### Secondary Dataset: SciBench Physics Subset
- **Why**: Physics-specific benchmark, tests domain hypothesis
- **Coverage**: College-level physics problems
- **Difficulty**: Baseline CoT only achieves 6.8% accuracy
- **Advantage**: Tests if narrative helps on hardest physics problems

#### Optional Dataset: GSM8K (if time permits)
- **Why**: Standard math reasoning benchmark
- **Purpose**: Test if narrative helps on simpler math problems
- **Advantage**: Establishes generalization beyond graduate-level problems

### Baselines

#### Must-Have Baselines
1. **Zero-shot**: Direct question → answer
   - Simplest baseline
   - Shows model's innate capability

2. **Zero-shot CoT**: "Let's think step by step"
   - Standard CoT method
   - Widely used comparison point

3. **Story of Thought (SoT)**: 3-step narrative generation
   - Primary comparison method
   - Establishes whether we can reproduce published results

#### Comparison Rationale
- Zero-shot shows baseline performance without reasoning
- Zero-shot CoT shows standard reasoning capability
- SoT shows narrative-based reasoning capability
- Novel variations test if we can improve on or simplify SoT

### Evaluation Metrics

#### Primary Metrics

**1. Accuracy** (% correct answers)
- **Why**: Standard across all papers, enables comparison
- **Computation**: (Correct answers / Total questions) × 100%
- **Reporting**: Overall + per-domain breakdown

**2. Per-Domain Accuracy** (Physics, Chemistry, Math)
- **Why**: Tests H1a (domain specificity hypothesis)
- **Computation**: Accuracy within each subject
- **Analysis**: Compare improvement ratios across domains

#### Secondary Metrics

**3. Token Usage** (cost-effectiveness)
- **Why**: Practical deployment consideration
- **Computation**: Total tokens per method / number of problems
- **Analysis**: Performance vs. cost trade-off

**4. Error Analysis** (qualitative)
- **Why**: Understand failure modes
- **Categories**:
  - Factual errors (wrong facts)
  - Reasoning errors (wrong logic)
  - Format errors (parsing failures)
  - Calculation errors (arithmetic mistakes)

#### Statistical Analysis

**Statistical Tests**:
- **Paired t-test**: Compare methods on same questions
- **Significance level**: α = 0.05
- **Effect size**: Cohen's d for practical significance
- **Confidence intervals**: 95% CI for all accuracy metrics

**Sample Size Considerations**:
- JEEBench: 515 total, ~170 per domain (adequate)
- SciBench Physics: ~200 problems (adequate)
- Statistical power: >0.8 for detecting 10% improvement

### Models

#### Primary Model: GPT-4 (gpt-4)
- **Why**: Used in SoT paper, enables direct comparison
- **Advantage**: Strong baseline performance
- **Cost**: ~$1.25/M input tokens, ~$10/M output tokens

#### Secondary Model: GPT-4.1 (gpt-4-turbo) or GPT-5 (if available)
- **Why**: Test if narrative helps with newest models
- **Advantage**: Better reasoning capabilities
- **Expected**: May show different narrative sensitivity

#### Optional: GPT-3.5-turbo (if budget permits)
- **Why**: Test if narrative helps smaller models
- **Advantage**: Much cheaper, tests scalability
- **Expected**: May benefit less from narrative (per literature)

### Expected Outcomes

#### Scenario 1: Hypothesis Supported
- Story-based CoT shows 5-15% absolute improvement on physics problems
- Improvement is larger on physics than chemistry/math
- Statistical significance: p < 0.05
- **Conclusion**: Narrative helps physics reasoning

#### Scenario 2: Hypothesis Partially Supported
- Story-based CoT improves overall but not specifically on physics
- Improvement is similar across domains
- **Conclusion**: Narrative helps reasoning generally, not physics-specific

#### Scenario 3: Hypothesis Not Supported
- Story-based CoT shows no significant improvement
- Or improvement only on non-physics problems
- **Conclusion**: Need to investigate why narrative doesn't help physics

#### Scenario 4: Novel Findings
- Simpler narrative variants match or exceed SoT performance
- Specific narrative techniques drive all improvements
- **Conclusion**: Identify key narrative components

## Timeline and Milestones

### Total Time Budget: 3 hours (180 minutes)

#### Milestone 1: Environment Setup (15 min) ✓
- Create virtual environment
- Install dependencies
- Verify API access
- Load and validate datasets

#### Milestone 2: Baseline Implementation (20 min)
- Implement zero-shot, zero-shot CoT
- Test on 10 examples
- Verify evaluation pipeline
- **Checkpoint**: Baseline accuracy ~30-40% on JEEBench

#### Milestone 3: SoT Implementation (40 min)
- Implement 3-step SoT process
- Test on 20 examples
- Debug any issues
- **Checkpoint**: Match SoT paper ballpark results

#### Milestone 4: Novel Variations (30 min)
- Implement 2-3 narrative variations
- Test on subset
- Select best variants
- **Checkpoint**: Identify promising approaches

#### Milestone 5: Full Evaluation (40 min)
- Run all methods on complete datasets
- Collect results
- Generate tables and plots
- **Checkpoint**: Complete results matrix

#### Milestone 6: Analysis & Documentation (35 min)
- Statistical analysis
- Error analysis
- Create REPORT.md with findings
- Create README.md
- **Checkpoint**: Complete documentation

**Buffer**: 20 minutes for debugging and unexpected issues

## Potential Challenges and Mitigation

### Challenge 1: API Rate Limits
**Risk**: Hit OpenAI rate limits during evaluation
**Mitigation**:
- Use exponential backoff retry logic
- Implement caching for repeated calls
- Monitor rate limit headers

### Challenge 2: Inconsistent Model Responses
**Risk**: High variance in model outputs affects results
**Mitigation**:
- Set temperature=0 for deterministic responses
- Use consistent random seeds
- Consider self-consistency (majority vote) if variance is high

### Challenge 3: Dataset Format Issues
**Risk**: Difficulty parsing answers from datasets
**Mitigation**:
- Inspect dataset structure first
- Write robust answer extraction functions
- Handle multiple answer formats

### Challenge 4: Reproducing SoT Results
**Risk**: Cannot match SoT paper performance
**Mitigation**:
- Use exact prompts from paper (Appendix C if available)
- Test on small subset first
- Document any deviations
- If still can't reproduce, report as limitation

### Challenge 5: Time Constraints
**Risk**: Not enough time for full evaluation
**Mitigation**:
- Prioritize JEEBench physics subset
- Use smaller sample if needed (but document this)
- Focus on core comparisons (zero-shot, CoT, SoT)
- Skip optional baselines if time runs short

### Challenge 6: Budget Constraints
**Risk**: API costs exceed $100 budget
**Mitigation**:
- Estimate costs before running: ~500 problems × 3 methods × 1000 tokens × $0.01/1K = ~$15-30
- Should be well within budget
- Monitor costs during execution
- Use GPT-3.5-turbo for testing/debugging (cheaper)

## Success Criteria

### Minimum Success
✓ Implement baselines (zero-shot, CoT)
✓ Implement Story of Thought methodology
✓ Evaluate on JEEBench (at least 100 problems)
✓ Report accuracy comparison
✓ Document findings in REPORT.md

### Good Success
✓ All of above, plus:
✓ Full evaluation on complete JEEBench (515 problems)
✓ Per-domain analysis (Physics, Chemistry, Math)
✓ Statistical significance testing
✓ Error analysis with examples

### Excellent Success
✓ All of above, plus:
✓ Reproduce SoT paper results (within ±3%)
✓ Test novel narrative variations
✓ Evaluate on SciBench physics subset
✓ Identify specific narrative techniques that help
✓ Actionable insights for future work

## Resource Planning

### Computational Resources
- **Required**: CPU sufficient (API-based research)
- **Memory**: 8-16GB (for loading datasets)
- **Storage**: <1GB (datasets already downloaded)
- **No GPU required**: All inference via API

### Budget Estimate

#### API Costs (Conservative Estimate)
- **JEEBench**: 515 problems × 4 methods × 1000 tokens avg × $0.015/1K = ~$30
- **SciBench**: 200 problems × 2 methods × 1000 tokens × $0.015/1K = ~$6
- **Testing/debugging**: ~$10
- **Buffer for retries**: ~$15
- **Total Estimated**: $60-70

Well within $100 budget with significant safety margin.

### Python Dependencies
```
openai>=1.0.0          # OpenAI API
anthropic>=0.7.0       # Anthropic API (optional)
datasets>=2.14.0       # HuggingFace datasets
numpy>=1.24.0          # Numerical computing
pandas>=2.0.0          # Data analysis
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical visualization
scikit-learn>=1.3.0    # Statistical tests
scipy>=1.11.0          # Statistical functions
tqdm>=4.65.0           # Progress bars
```

## Experimental Protocol Details

### Prompt Templates

#### Zero-Shot
```
Question: {question}

Answer:
```

#### Zero-Shot CoT
```
Question: {question}

Let's think step by step.
```

#### Story of Thought (3-step)
```
# Step 1: Question Clarification
Break down the following question into its core components:
{question}

# Step 2: Narrative Generation
Create a narrative using the following techniques:
- Progressive Disclosure: Gradually reveal information
- Analogy: Draw parallels to familiar concepts
- Analogical Reasoning: Apply reasoning from similar situations
- Metaphor: Use metaphorical explanations
- Branching: Explore alternative explanations

# Step 3: Problem Solving
Using the narrative above, solve the problem:
{question}
```

#### Memory Recall Variation
```
Question: {question}

Think about this like recalling a memory. Have you encountered similar physics
problems before? What do you remember about how to approach this type of
situation? Tell a story about solving a similar problem, then apply that
approach here.
```

#### Story Explanation Variation
```
Question: {question}

Explain this problem as if you're telling a story to help someone understand
the physics concepts. What's the narrative that makes this situation clear?
Use the story to guide you to the answer.
```

### Answer Extraction
- Parse model output for final answer
- Handle multiple formats (numeric, multiple choice, free-form)
- Implement fuzzy matching for near-matches
- Log extraction failures for analysis

### Reproducibility
- Random seed: 42 (for any stochastic operations)
- Temperature: 0.0 (deterministic responses)
- Model versions: Log exact model IDs
- Timestamp all experiments
- Save all raw model outputs

## Validation Plan

### Code Validation
- Test on 5 examples manually
- Verify answer extraction works correctly
- Check that prompts format properly
- Ensure no data leakage

### Result Validation
- Sanity check: Zero-shot should be lowest
- CoT should improve over zero-shot
- Results should be in reasonable range (20-50% accuracy)
- No unexpected 100% or 0% accuracies

### Statistical Validation
- Check test assumptions (normality, independence)
- Use appropriate tests for data type
- Report effect sizes, not just p-values
- Correct for multiple comparisons if testing many hypotheses

## Next Steps After This Research

### Immediate Follow-Ups
1. **Deeper Error Analysis**: Categorize errors by physics concept
2. **Narrative Component Ablation**: Test each narrative technique individually
3. **Human Evaluation**: Compare narrative quality to standard CoT
4. **Cross-Model Validation**: Test on Llama, Claude, Gemini

### Alternative Approaches
1. **Fine-Tuning**: Train models on narrative reasoning examples
2. **Retrieval Augmentation**: Add physics knowledge to narratives
3. **Multi-Agent**: Multiple LLMs generate and vote on narratives
4. **Interactive Narrative**: Multi-turn dialogue for problem solving

### Broader Extensions
1. **Other Domains**: Test narrative reasoning on code, law, medicine
2. **Multimodal**: Add diagrams and equations to narratives
3. **Automated Narrative Generation**: Train model to create optimal narratives
4. **Narrative Theory**: Connect to cognitive science and education research

## Open Questions

1. **Why does narrative help?** Information organization? Knowledge retrieval? Mental model building?
2. **What makes a "good" narrative?** Length? Coherence? Specific techniques?
3. **Can models learn to generate narratives?** Training vs. prompting
4. **Domain specificity**: Why might physics benefit more than other subjects?
5. **Transferability**: Do narratives that help one model help others?

## References

1. Sadiri Javadi et al. (2024). "Can Stories Help LLMs Reason? Curating Information Space Through Narrative." arXiv:2410.19221

2. Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." arXiv:2201.11903

3. Wang et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." arXiv:2203.11171

4. Physics Reasoner (2024). "Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems." arXiv:2412.13791

5. Arora et al. (2023). "Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For JEEBench." EMNLP.

---

**Plan Status**: Ready to execute
**Next Phase**: Environment setup and baseline implementation
**Estimated Completion**: 3 hours from start
