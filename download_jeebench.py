#!/usr/bin/env python3
"""Download JEEBench dataset from HuggingFace"""

from datasets import load_dataset
import os

print("Downloading JEEBench dataset...")
try:
    # Try loading the dataset without specifying a split
    jeebench = load_dataset("daman1209arora/jeebench", split="test")

    # Save to disk
    jeebench.save_to_disk("jeebench")

    print(f"✓ JEEBench dataset downloaded: {len(jeebench)} examples")

    # Show a sample
    print("\nSample question:")
    print(jeebench[0])

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")
    try:
        jeebench = load_dataset("daman1209arora/jeebench")
        jeebench.save_to_disk("jeebench_full")
        print("✓ Dataset downloaded using alternative method")
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
