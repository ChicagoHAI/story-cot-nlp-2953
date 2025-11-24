# Datasets for Story CoT Research

This directory contains datasets for the "Story CoT" research project investigating narrative-based chain-of-thought reasoning in language models. **Data files are NOT committed to git due to size**. Follow the download instructions below.

## Overview

| Dataset | Source | Size | Task | Status |
|---------|--------|------|------|--------|
| GSM8K | HuggingFace (openai/gsm8k) | 7.5K train, 1.3K test | Math word problems | ✓ Downloaded |
| SciBench | HuggingFace (xw27/scibench) | 692 problems | College physics/chem/math | ✓ Downloaded |
| JEEBench | HuggingFace (daman1209arora/jeebench) | 515 problems | Physics/chem/math (JEE Advanced) | ✓ Downloaded |
| GPQA | HuggingFace (Idavidrein/gpqa) | 198 questions (Diamond) | Graduate-level science | ⚠ Gated (requires auth) |

## Dataset Descriptions

### 1. GSM8K (Grade School Math 8K)

**Overview:**
- High quality linguistically diverse grade school math word problems
- Created by OpenAI for testing multi-step mathematical reasoning
- MIT License

**Statistics:**
- Training: 7,473 examples
- Test: 1,319 examples
- Average steps per solution: 2-8
- Difficulty: Grade school level

**Download Instructions:**

```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k")
```

**Automated download:**
```bash
python3 download_datasets.py
```

**Loading the Dataset:**

```python
from datasets import load_from_disk
gsm8k = load_from_disk("datasets/gsm8k")

# Access examples
train_example = gsm8k['train'][0]
print(train_example['question'])
print(train_example['answer'])
```

**Sample Data:**
```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\\n#### 72"
}
```

**Use Case for Story CoT:**
- Baseline for mathematical reasoning
- Test if story-based CoT improves performance on word problems
- Compare with standard CoT approaches

---

### 2. SciBench (College-level Science Problems)

**Overview:**
- Benchmark for college-level scientific problem solving
- Sourced from instructional textbooks
- Domains: Mathematics, Chemistry, Physics
- All open-ended, free-response questions

**Statistics:**
- Total: 692 problems
- Requires multi-step reasoning
- Domain knowledge essential
- Best LLM performance: ~49% (as of 2024)

**Download Instructions:**

```python
from datasets import load_dataset
scibench = load_dataset("xw27/scibench")
scibench.save_to_disk("datasets/scibench")
```

**Loading the Dataset:**

```python
from datasets import load_from_disk
scibench = load_from_disk("datasets/scibench")

# Access examples
example = scibench['train'][0]
```

**Use Case for Story CoT:**
- Test narrative-based reasoning on physics problems (per hypothesis)
- Compare with standard CoT on complex science problems
- Evaluate if story context helps with domain knowledge retrieval

---

### 3. JEEBench (IIT JEE Advanced Problems)

**Overview:**
- 515 challenging pre-engineering problems from IIT JEE-Advanced exam
- One of the most competitive engineering entrance exams globally
- Subjects: Physics, Mathematics, Chemistry
- Years: 2016-2023

**Statistics:**
- Total: 515 problems
- Physics, Chemistry, Mathematics
- Multiple question types: Single-correct, Multi-correct, Integer, Numeric
- Best GPT-4 performance: ~39% (with CoT + Self-Consistency)

**Download Instructions:**

```python
from datasets import load_dataset
jeebench = load_dataset("daman1209arora/jeebench", split="test")
jeebench.save_to_disk("datasets/jeebench")
```

**Automated download:**
```bash
python3 download_jeebench.py
```

**Loading the Dataset:**

```python
from datasets import load_from_disk
jeebench = load_from_disk("datasets/jeebench")

# Access examples
example = jeebench[0]
print(f"Subject: {example['subject']}")
print(f"Question: {example['question']}")
print(f"Correct answer: {example['gold']}")
```

**Sample Data:**
```json
{
  "subject": "phy",
  "description": "JEE Adv 2016 Paper 1",
  "gold": "B",
  "type": "MCQ",
  "question": "In a historical experiment to determine Planck's constant..."
}
```

**Use Case for Story CoT:**
- **PRIMARY EVALUATION DATASET** - Used in SoT paper
- Test story-based reasoning on extremely challenging problems
- Physics problems directly relevant to hypothesis
- Compare performance across subjects (physics, chem, math)

---

### 4. GPQA (Graduate-level Problem-solving QA)

**Overview:**
- Graduate-level "Google-proof" Q&A benchmark
- Expert-crafted multiple-choice questions
- Domains: Biology, Physics, Chemistry
- Diamond subset: 198 high-agreement questions

**Statistics:**
- Total: 448 questions (198 in Diamond subset)
- Expert PhD accuracy: 65%
- Non-expert with Google: 34%
- GPT-4 baseline: 39%

**Download Instructions:**

**⚠ Note: This dataset is GATED and requires authentication**

1. Create HuggingFace account and get access token
2. Authenticate:
```bash
huggingface-cli login
```

3. Download:
```python
from datasets import load_dataset
gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
gpqa.save_to_disk("datasets/gpqa_diamond")
```

**Loading the Dataset:**

```python
from datasets import load_from_disk
gpqa = load_from_disk("datasets/gpqa_diamond")

# Access examples
example = gpqa['train'][0]
```

**Use Case for Story CoT:**
- **PRIMARY EVALUATION DATASET** - Used in SoT paper
- Graduate-level difficulty ideal for testing advanced reasoning
- Multiple domains for comprehensive evaluation
- Direct comparison with SoT paper results

---

## Quick Start

### Download All Datasets

```bash
# Run the automated download script
cd datasets
python3 download_datasets.py
python3 download_jeebench.py
```

This will download GSM8K, SciBench, and JEEBench. GPQA requires manual authentication.

### Verify Downloads

```bash
ls -lh datasets/
```

You should see directories:
- `gsm8k/`
- `scibench/`
- `jeebench/`

### Load and Inspect

```python
from datasets import load_from_disk

# Load a dataset
gsm8k = load_from_disk("datasets/gsm8k")

# Inspect
print(f"Train size: {len(gsm8k['train'])}")
print(f"Test size: {len(gsm8k['test'])}")
print(f"Sample: {gsm8k['train'][0]}")
```

##Recommended Datasets for Story CoT Experiments

Based on the literature review and the Story of Thought (SoT) paper:

1. **Primary**: GPQA (Diamond) - Direct comparison with SoT paper results
2. **Primary**: JEEBench - Physics problems as mentioned in hypothesis
3. **Secondary**: GSM8K - Baseline math reasoning benchmark
4. **Secondary**: SciBench - Additional physics evaluation

## Common Issues and Solutions

### Issue: GPQA authentication error
**Solution**:
1. Visit https://huggingface.co/datasets/Idavidrein/gpqa
2. Request access
3. Run `huggingface-cli login` with your token

### Issue: Out of memory during download
**Solution**: Download datasets one at a time rather than using the batch script

### Issue: Dataset appears corrupted
**Solution**: Delete the dataset directory and re-download:
```bash
rm -rf datasets/gsm8k
python3 download_datasets.py
```

## Citation

If using these datasets, please cite the original papers:

**GSM8K:**
```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

**JEEBench:**
```bibtex
@inproceedings{arora2023jeebench,
  title={Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models},
  author={Arora, Daman and Singh, Himanshu and Mausam},
  booktitle={EMNLP},
  year={2023}
}
```

**GPQA:**
```bibtex
@inproceedings{rein2024gpqa,
  title={GPQA: A Graduate-Level Google-Proof Q&A Benchmark},
  author={Rein, David and Hou, Betty Li and Stickland, Asa Cooper and others},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
```

**SciBench:**
```bibtex
@inproceedings{wang2023scibench,
  title={SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models},
  author={Wang, Xiaoxuan and Hu, Ziniu and Lu, Pan and others},
  booktitle={ICML},
  year={2024}
}
```

## License Information

- **GSM8K**: MIT License
- **JEEBench**: Check repository for license
- **GPQA**: Gated access, check terms of use
- **SciBench**: Check repository for license

## Notes

- Dataset files are excluded from git (see .gitignore)
- Total disk space needed: ~100MB (without GPQA)
- Download time: 2-5 minutes (depends on connection)
- All datasets are in HuggingFace Dataset format for easy loading
