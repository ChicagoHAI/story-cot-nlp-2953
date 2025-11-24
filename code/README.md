# Code Repositories for Story CoT Research

This directory contains cloned repositories with implementations of chain-of-thought and related reasoning methods. These provide baselines and reference implementations for the Story CoT research project.

## Cloned Repositories

### 1. Auto-CoT (Automatic Chain of Thought)

**Location**: `code/auto-cot/`

**Repository**: https://github.com/amazon-science/auto-cot

**Purpose**: Official implementation of "Automatic Chain of Thought Prompting in Large Language Models" (ICLR 2023)

**Key Features**:
- Automatically constructs demonstrations with question clustering and demonstration sampling
- Eliminates manual design of CoT prompts
- Matches or exceeds performance of manual CoT design on GPT-3
- Evaluated on 10 public benchmark reasoning tasks

**Usage**:
```bash
cd code/auto-cot

# Construct demonstrations
python run_demo.py --task multiarith --pred_file log/multiarith_zero_shot_cot.log --demo_save_dir demos/multiarith

# Run inference
python run_inference.py --dataset multiarith --demo_path demos/multiarith --output_dir experiment/multiarith
```

**Dependencies**:
- Python >= 3.8
- PyTorch 1.8.2+cu111
- torchtext 0.9.2

**Relevance to Story CoT**:
- Provides automated baseline for comparison
- Can adapt demonstration construction methods
- Good reference for prompt engineering techniques

---

### 2. Tree of Thoughts (ToT)

**Location**: `code/tree-of-thought-llm/`

**Repository**: https://github.com/princeton-nlp/tree-of-thought-llm

**Purpose**: Official implementation of "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (NeurIPS 2023)

**Key Features**:
- Enables LLMs to explore multiple reasoning paths
- Deliberate decision-making by considering multiple thought branches
- Self-evaluation of progress through intermediate thoughts
- Lookahead and backtracking search algorithms

**Usage**:
```bash
cd code/tree-of-thought-llm

# Run experiments
python run.py --task game24 --method bfs --model gpt-4
```

**Relevance to Story CoT**:
- Alternative structured reasoning approach
- Baseline for comparison on complex problems
- Shows performance on tasks requiring exploration
- Used in SoT paper as comparison method

---

### 3. Chain-of-Thought Hub

**Location**: `code/chain-of-thought-hub/`

**Repository**: https://github.com/FranxYao/chain-of-thought-hub

**Purpose**: Benchmarking framework for evaluating LLMs' complex reasoning abilities with CoT prompting

**Key Features**:
- Comprehensive benchmark suite
- Multiple datasets and evaluation metrics
- Standardized CoT evaluation protocols
- Support for various LLM APIs

**Usage**:
```bash
cd code/chain-of-thought-hub

# Run benchmarks
python run_benchmark.py --model gpt-3.5-turbo --dataset gsm8k
```

**Relevance to Story CoT**:
- Standardized evaluation framework
- Can integrate Story CoT methods
- Provides baseline CoT results
- Good for systematic comparison

---

## Recommended Usage Pattern

### For Baseline Experiments

1. **Auto-CoT**: Use for automated CoT baseline
   ```python
   # Generate demonstrations automatically
   from auto_cot import construct_demos
   demos = construct_demos(dataset="physics_problems")
   ```

2. **Tree of Thoughts**: Use for structured reasoning baseline
   ```python
   # Apply ToT to physics problems
   from tot import tree_of_thoughts
   result = tree_of_thoughts(problem, method="bfs")
   ```

3. **CoT Hub**: Use for standardized evaluation
   ```python
   # Benchmark against standard CoT
   from cot_hub import evaluate
   results = evaluate(model, dataset="GPQA", method="cot")
   ```

### For Story CoT Implementation

These repositories provide reference implementations that can be adapted:

- **Prompt engineering patterns** from Auto-CoT
- **Multi-path reasoning** concepts from ToT
- **Evaluation frameworks** from CoT Hub

## Installation Instructions

### Auto-CoT

```bash
cd code/auto-cot
pip install torch==1.8.2+cu111 torchtext==0.9.2
pip install -r requirements.txt  # if available
```

### Tree of Thoughts

```bash
cd code/tree-of-thought-llm
pip install openai  # or other LLM API clients
pip install -r requirements.txt  # if available
```

### Chain-of-Thought Hub

```bash
cd code/chain-of-thought-hub
pip install -r requirements.txt
```

## Key Files to Explore

### Auto-CoT
- `run_demo.py`: Demonstration construction logic
- `run_inference.py`: Inference with constructed demos
- `api.py`: LLM API interactions
- `try_cot.ipynb`: Jupyter notebook tutorial

### Tree of Thoughts
- `src/tot/methods/`: BFS and DFS search implementations
- `src/tot/tasks/`: Task-specific implementations
- `src/tot/prompts/`: Prompt templates

### Chain-of-Thought Hub
- `benchmarks/`: Various reasoning benchmarks
- `prompts/`: CoT prompt templates
- `evaluation/`: Evaluation metrics and scripts

## Integration with Story CoT

These repositories can be used as:

1. **Baselines**: Compare Story CoT against Auto-CoT and ToT
2. **Code reference**: Learn prompt engineering and LLM interaction patterns
3. **Evaluation framework**: Use CoT Hub for standardized evaluation
4. **Components**: Adapt demonstration construction or evaluation methods

## Common Modifications Needed

To integrate with Story CoT experiments:

1. **Adapt prompts**: Modify to include narrative elements
2. **Custom datasets**: Point to GPQA, JEEBench, etc.
3. **Evaluation metrics**: Ensure consistency with SoT paper
4. **LLM selection**: Configure for models used in experiments (Llama 3, GPT-4, etc.)

## Citation Information

If using these codebases, cite the original papers:

**Auto-CoT:**
```bibtex
@inproceedings{zhang2023automatic,
  title={Automatic Chain of Thought Prompting in Large Language Models},
  author={Zhang, Zhuosheng and Zhang, Aston and Li, Mu and Smola, Alex},
  booktitle={ICLR},
  year={2023}
}
```

**Tree of Thoughts:**
```bibtex
@inproceedings{yao2023tree,
  title={Tree of Thoughts: Deliberate Problem Solving with Large Language Models},
  author={Yao, Shunyu and Yu, Dian and Zhao, Jeffrey and Shafran, Izhak and Griffiths, Thomas L and Cao, Yuan and Narasimhan, Karthik},
  booktitle={NeurIPS},
  year={2023}
}
```

**CoT Hub:**
```bibtex
@article{yao2022cot,
  title={Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance},
  author={Yao, Shunyu and others},
  year={2022}
}
```

## Notes

- These repositories are reference implementations and may need adaptation
- Check each repo's LICENSE file for usage terms
- Some repos may require API keys (OpenAI, Anthropic, etc.)
- Update dependencies as needed for compatibility with your environment

## Troubleshooting

### Issue: Import errors
**Solution**: Ensure you're in the correct directory and have installed dependencies

### Issue: API rate limits
**Solution**: Implement rate limiting and retry logic

### Issue: CUDA errors
**Solution**: Check PyTorch/CUDA compatibility, may need different PyTorch version

## Additional Resources

- **SoT Paper Implementation**: Not found in public repos yet - will need to implement from paper
- **Graph of Thoughts**: https://github.com/spcl/graph-of-thoughts
- **Self-Consistency**: Implemented in most CoT frameworks

## Next Steps

1. Review each repository's code structure
2. Identify reusable components for Story CoT
3. Set up environment and test installations
4. Adapt prompts for narrative-based reasoning
5. Integrate with datasets (GPQA, JEEBench, etc.)
