# Literature Review: Story CoT

## Research Area Overview

This literature review examines chain-of-thought (CoT) reasoning in large language models, with particular focus on narrative-based approaches. The central research question is whether training/prompting LLMs to use story-like, narrative-based reasoning improves performance compared to standard CoT methods, especially on physics and complex reasoning tasks.

The field has evolved from simple prompting techniques to sophisticated structured reasoning approaches, with recent work exploring how narrative structures—long recognized in education and science communication—can enhance LLM reasoning capabilities.

---

## Key Papers

### 1. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Citation**: Wei et al., 2022 (arXiv:2201.11903)

**Key Contribution**: Foundational paper demonstrating that generating intermediate reasoning steps (chain of thought) significantly improves LLM performance on complex reasoning tasks.

**Methodology**:
- Few-shot prompting with exemplars containing reasoning chains
- Tested on arithmetic, commonsense, and symbolic reasoning tasks
- Evaluated with models up to 540B parameters

**Datasets Used**:
- GSM8K (math word problems)
- SVAMP, ASDiv, AQuA, MAWPS (arithmetic)
- CommonsenseQA, StrategyQA (commonsense reasoning)
- Date understanding, sports understanding (symbolic reasoning)

**Results**:
- 540B model achieved state-of-the-art 58% on GSM8K (with just 8 exemplars)
- Gains emerge with model scale (>100B parameters)
- Minimal benefit for small models

**Code Available**: Yes (various third-party implementations)

**Relevance to Our Research**: Establishes baseline CoT method. Our Story CoT approach builds on this by adding narrative structure to the reasoning chain.

---

### 2. Can Stories Help LLMs Reason? Curating Information Space Through Narrative

**Citation**: Sadiri Javadi et al., 2024 (arXiv:2410.19221)

**Key Contribution**: **MOST DIRECTLY RELEVANT** - Introduces Story of Thought (SoT), a novel approach that integrates narrative structures into prompting for problem solving.

**Methodology**:
- Three-step approach:
  1. Question Clarification: Break down question into core components
  2. Narrative Generation: Create narrative using 5 techniques (Progressive Disclosure, Branching, Analogy, Analogical Reasoning, Metaphor)
  3. Problem Solving: Use narrative to solve the task

**Datasets Used**:
- GPQA (Diamond subset, 198 questions)
- JEEBench (515 problems)

**Baselines Compared**:
- Zero-shot, Zero-shot CoT
- Tree of Thoughts, Graph of Thoughts
- Analogical Reasoning (3-shot)

**Results**:
- **GPQA**: Llama 3 70B with SoT achieved 51.01% (vs 39.5% zero-shot)
  - GPT-4: 48.98% with SoT (vs 34.7% zero-shot) - 41% relative improvement
- **JEEBench**: Llama 3 70B with SoT achieved 0.453 aggregate score
  - Outperformed previous SOTA (GPT-4+CoT+SC@8: 0.389)
  - Best performance on 6/8 models tested

**Key Findings**:
- Narrative techniques help LLMs identify and organize relevant information
- Progressive Disclosure and Analogy most consistently utilized
- Larger models benefit more from narrative generation
- Biology problems showed most improvement, then Chemistry
- Combination of narrative techniques more effective than any single technique

**Code Available**: Not publicly released yet

**Relevance to Our Research**: **DIRECTLY ADDRESSES OUR HYPOTHESIS**. Provides concrete methodology, results, and evidence that story-based reasoning improves LLM performance on physics and science problems.

---

### 3. Self-Consistency Improves Chain of Thought Reasoning in Language Models

**Citation**: Wang et al., 2022 (arXiv:2203.11171)

**Key Contribution**: Proposes self-consistency decoding—sample diverse reasoning paths and select the most consistent answer by marginalization.

**Methodology**:
- Generate multiple CoT reasoning paths (via sampling with temperature > 0)
- Aggregate by selecting most frequent answer
- "Self-consistency" = diverse paths should converge to correct answer

**Datasets Used**:
- Arithmetic: GSM8K, SVAMP, AQuA, etc.
- Commonsense: CommonsenseQA, StrategyQA
- Symbolic: Date, Sports understanding

**Results**:
- Consistent improvements over greedy decoding CoT
- GSM8K: 74.4% (vs 56.5% with greedy CoT) using UL2-20B
- Larger improvements on more complex tasks

**Code Available**: Implementations in various CoT frameworks

**Relevance to Our Research**: Important baseline and potential enhancement technique. Could combine with Story CoT (generate multiple narratives, select most consistent).

---

### 4. Multimodal Chain-of-Thought Reasoning in Language Models

**Citation**: Zhang et al., 2023 (arXiv:2302.00923)

**Key Contribution**: Extends CoT to multimodal (text + vision) settings with two-stage framework separating rationale generation and answer inference.

**Methodology**:
- Stage 1: Rationale generation (using both text and image)
- Stage 2: Answer inference (leveraging generated rationale)
- Two-stage separation prevents vision features from dominating

**Datasets Used**:
- ScienceQA (multimodal science questions)

**Results**:
- Achieved new SOTA on ScienceQA
- Showed importance of separating reasoning generation from answer prediction

**Relevance to Our Research**: Demonstrates value of structured multi-stage reasoning. SoT also uses stages (clarification → narrative → solving). Could inform multimodal extensions of Story CoT.

---

### 5. Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems

**Citation**: Physics Reasoner, 2024 (arXiv:2412.13791)

**Key Contribution**: Shows that LLMs struggle with physics problems even with advanced methods; proposes knowledge-augmented approach.

**Methodology**:
- Knowledge retrieval from physics knowledge base
- Integration with reasoning process
- Testing with various CoT methods

**Datasets Used**:
- SciBench (physics section)
- JEE Advanced physics problems

**Results**:
- GPT-3.5-turbo: Only 6.8% accuracy on SciBench physics (with CoT)
- **Critical finding**: Standard CoT insufficient for physics reasoning
- Knowledge augmentation provides improvements

**Relevance to Our Research**: **Directly relevant to physics problem-solving hypothesis**. Shows gap that Story CoT might address. Physics problems are challenging benchmark for testing narrative reasoning benefits.

---

### 6. Automatic Chain of Thought Prompting in Large Language Models

**Citation**: Zhang et al., 2022 (arXiv:2210.03493)

**Key Contribution**: Automates CoT prompt design through question clustering and demonstration sampling.

**Methodology**:
- Cluster questions by similarity
- Sample representative question from each cluster
- Generate rationale with Zero-Shot-CoT
- Use as demonstrations for few-shot prompting

**Datasets Used**:
- 10 benchmark tasks (arithmetic, commonsense, symbolic)
- Includes GSM8K, MultiArith, CommonsenseQA, etc.

**Results**:
- Matches or exceeds manual CoT design on GPT-3
- Eliminates manual effort in demonstration construction
- More robust to demonstration variance

**Code Available**: Yes (github.com/amazon-science/auto-cot)

**Relevance to Our Research**: Provides automated baseline approach. Could adapt Auto-CoT's demonstration construction for Story CoT (auto-generate narrative exemplars).

---

## Common Methodologies

Across the reviewed papers, several common approaches emerge:

### 1. **Few-Shot Prompting with Exemplars**
- Used in: CoT, Auto-CoT, Analogical Reasoning
- Provide 3-8 examples with reasoning steps
- Model learns to mimic reasoning pattern

### 2. **Multi-Stage Reasoning**
- Used in: SoT, Multimodal-CoT
- Separate problem understanding from answer generation
- Intermediate representations enhance reasoning

### 3. **Sampling and Aggregation**
- Used in: Self-Consistency, Tree of Thoughts
- Generate multiple reasoning paths
- Select answer via voting or consistency

### 4. **Structured Reasoning**
- Used in: Tree of Thoughts, Graph of Thoughts, SoT
- Impose explicit structure (tree, graph, narrative)
- Explore multiple paths or organize information

### 5. **Knowledge Augmentation**
- Used in: Physics Reasoner
- Retrieve relevant domain knowledge
- Integrate with reasoning process

---

## Standard Baselines

Based on the literature, the following baselines are commonly used:

| Baseline | Description | When to Use | Typical Performance (GPT-4 on GPQA) |
|----------|-------------|-------------|-------------------------------------|
| Zero-shot | Direct prompting without examples | Simplest baseline | ~35% |
| Zero-shot CoT | "Let's think step by step" | Standard reasoning baseline | ~36% |
| Few-shot CoT | Manual exemplars with reasoning | Common comparison point | ~40% |
| Self-Consistency | Sample multiple CoT, aggregate | Strong baseline | ~42-45% |
| Auto-CoT | Automated demonstration construction | Automated baseline | ~41% |
| Tree of Thoughts | Explore multiple reasoning trees | Structured reasoning baseline | ~42% |
| **Story of Thought** | Narrative-based reasoning | Our approach | **~49% (GPT-4)** |

---

## Evaluation Metrics

Papers consistently use the following metrics:

### Primary Metrics

**Accuracy**: Percentage of correct answers
- Most common metric across all papers
- Used for multiple-choice (GPQA) and exact match (GSM8K)

**Aggregate Score**: Weighted average across problem types
- Used in JEEBench (single-correct, multi-correct, integer, numeric)
- Accounts for different difficulty levels

### Secondary Metrics

**BLEU/ROUGE/BERTScore**: Similarity to human explanations
- SoT paper uses to compare reasoning quality
- Measures explanation human-likeness

**Pass@k**: Percentage passing when generating k attempts
- Used in code generation and some reasoning tasks

**Token Efficiency**: Tokens used per problem
- Important for practical deployment
- Some methods (Sketch-of-Thought) optimize for this

---

## Datasets in the Literature

### Most Commonly Used Datasets

**Math Reasoning:**
- **GSM8K**: 8.5K grade school math problems (most popular)
- **SVAMP**: Math word problems with varying difficulty
- **AQuA**: Algebraic word problems

**Science Reasoning:**
- **GPQA**: Graduate-level science (biology, physics, chemistry)
- **JEEBench**: Engineering entrance exam problems
- **SciBench**: College-level science problems
- **ScienceQA**: Multimodal science questions

**Commonsense Reasoning:**
- **CommonsenseQA**: Multiple-choice commonsense questions
- **StrategyQA**: Questions requiring implicit reasoning

**Symbolic Reasoning:**
- **Date Understanding**: Temporal reasoning
- **Sports Understanding**: Domain-specific reasoning

### Dataset Usage in SoT Paper

- **Primary**: GPQA (Diamond), JEEBench
- **Rationale**: Graduate/advanced-level difficulty requires sophisticated reasoning
- **Coverage**: Physics, chemistry, biology, mathematics

---

## Gaps and Opportunities

Based on the literature review, several gaps and opportunities emerge:

### Gap 1: Limited Understanding of WHY Narrative Helps

**Current State**: SoT paper shows narrative improves performance but mechanisms unclear
- Is it better information organization?
- Enhanced knowledge retrieval?
- Improved mental model construction?

**Opportunity**: Conduct ablation studies and analysis to understand narrative's role

### Gap 2: No Direct Comparison of Different Narrative Styles

**Current State**: SoT uses one narrative approach (5 techniques combined)

**Opportunity**:
- Test "memory/story" framing vs. other narrative styles
- Compare first-person narrative vs. third-person
- Evaluate temporal vs. causal narrative structures

### Gap 3: Physics-Specific Evaluation Limited

**Current State**:
- SoT evaluated on mixed science problems
- Physics Reasoner shows standard CoT fails on physics

**Opportunity**: **DIRECTLY ADDRESSES OUR HYPOTHESIS**
- Focused evaluation on physics reasoning benchmarks
- Test if story-based CoT specifically helps physics modeling
- Compare physics vs. other domains

### Gap 4: Scalability and Automation

**Current State**: SoT requires multi-step prompting (3 API calls)

**Opportunity**:
- Single-step narrative prompting
- Automated narrative generation (like Auto-CoT)
- Cost-benefit analysis (tokens vs. accuracy)

### Gap 5: Training vs. Prompting

**Current State**: All methods use prompting only

**Opportunity**: **RELATES TO "TRAINING" HYPOTHESIS**
- Fine-tune models on narrative reasoning
- Train models to generate narratives internally
- Test if models can "learn" story-based reasoning

---

## Recommendations for Our Experiment

Based on this literature review, we recommend:

### Recommended Datasets

**Priority 1: GPQA (Diamond subset)**
- Reasoning: Direct comparison with SoT paper
- Difficulty: Graduate-level appropriate for testing sophisticated reasoning
- Coverage: Physics problems present

**Priority 2: JEEBench**
- Reasoning: Used in SoT paper, physics/chem/math split
- Difficulty: Extremely challenging (best model ~40%)
- Relevance: Physics problems match hypothesis

**Priority 3: SciBench (Physics subset)**
- Reasoning: Dedicated physics benchmark
- Difficulty: College-level physics
- Gap: Shows standard CoT fails (6.8% accuracy)

**Optional: GSM8K**
- Reasoning: Standard baseline, easy comparison
- Note: May be too simple for narrative benefits

### Recommended Baselines

**Must-have baselines:**
1. Zero-shot (simplest)
2. Zero-shot CoT (standard)
3. Few-shot CoT (common approach)
4. Story of Thought (main comparison)

**Good-to-have baselines:**
5. Self-Consistency CoT (strong baseline)
6. Auto-CoT (automated approach)
7. Tree of Thoughts (structured reasoning)

### Recommended Metrics

**Primary:**
- **Accuracy**: Standard, easy to compare

**Secondary:**
- **Per-domain breakdown**: Physics vs. Chemistry vs. Biology
- **Error analysis**: Categorize failure modes
- **Token usage**: Cost-effectiveness

**Optional:**
- **Explanation quality**: BLEU/BERTScore vs. human explanations
- **Consistency**: Agreement across multiple generations

### Methodological Considerations

**1. Prompt Design**
- Follow SoT paper's 3-step structure
- Test variations: "story" framing, "memory" framing, "narrative explanation"
- Include 5 narrative techniques (or ablate to find most effective)

**2. Model Selection**
- **Must test**: GPT-4 (for comparison with SoT paper)
- **Should test**: Llama 3 70B (best performer in SoT paper)
- **Optional**: Smaller models (test if narrative helps at smaller scale)

**3. Evaluation Protocol**
- Use same splits as SoT paper (GPQA Diamond, JEEBench full)
- Report both overall and per-domain results
- Include statistical significance testing

**4. Analysis**
- **Qualitative**: Examine narratives generated, compare to standard CoT
- **Quantitative**: Correlation between narrative quality and correctness
- **Domain-specific**: Test hypothesis that narrative helps more on physics

**5. Novel Contributions**
- Test "story of past experience" framing (memory/recollection)
- Evaluate on physics-specific benchmarks
- Investigate training-based approaches (if budget allows)

---

## Summary and Next Steps

### Key Takeaways

1. **CoT prompting** significantly improves LLM reasoning on complex tasks
2. **Narrative-based reasoning (SoT)** shows promising results, outperforming standard CoT
3. **Physics problems** are particularly challenging - opportunity for Story CoT
4. **Standardized benchmarks** (GPQA, JEEBench) enable comparison
5. **Multi-stage reasoning** and **information organization** appear important

### Research Questions to Address

1. Does "story of past experience" framing improve over general narrative?
2. Does narrative help more on physics vs. other domains?
3. Can we identify which narrative elements are most effective?
4. Does training (vs. prompting) with narratives improve performance?

### Immediate Next Steps

1. **Set up environment**: Install dependencies, configure LLM APIs
2. **Implement baselines**: Zero-shot, CoT, SoT following paper methodology
3. **Prepare datasets**: Load GPQA, JEEBench, ensure correct splits
4. **Initial experiments**: Run baselines on small subset, verify reproduction
5. **Develop Story CoT variants**: Implement "memory" and "experience" framings
6. **Full evaluation**: Run on complete datasets with statistical analysis

---

## References

1. Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903

2. Sadiri Javadi et al. (2024). Can Stories Help LLMs Reason? Curating Information Space Through Narrative. arXiv:2410.19221

3. Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171

4. Zhang et al. (2023). Multimodal Chain-of-Thought Reasoning in Language Models. arXiv:2302.00923

5. Physics Reasoner (2024). Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems. arXiv:2412.13791

6. Zhang et al. (2022). Automatic Chain of Thought Prompting in Large Language Models. arXiv:2210.03493
