"""
Data formatting utilities for converting dialogues to training format.
"""

import re
import random
from datasets import Dataset
import pandas as pd

from config import (
    IM_START, IM_END, SYSTEM, USER, ASSISTANT,
    TECHNIQUE_MENTION_PROB, DATA_PATH, DATA_SIZE, TRAIN_TEST_SPLIT,
    RANDOM_SEED
)
from prompts import get_system_prompt


def format_batch_full_chat(batch, prob=TECHNIQUE_MENTION_PROB):
    """
    Format a batch of dialogues into chat format for training.
    
    For each example in the batch, emit exactly one training string:
      <|im_start|>system
      [system prompt with optional CBT technique]
      <|im_end|>
      <|im_start|>user
      <client turn 1>
      <|im_end|>
      <|im_start|>assistant
      <counselor turn 1>
      <|im_end|>
      ...
    
    Args:
        batch: Dict containing 'dialogue_vi' and optionally 'cbt_technique_vi'
        prob: Probability of mentioning a specific technique
        
    Returns:
        Dict with 'text' key containing formatted conversations
    """
    out = []
    bs = len(batch['dialogue_vi'])
    
    for i in range(bs):
        dialogue = batch['dialogue_vi'][i]
        if not dialogue:
            continue

        # Decide whether to mention a specific technique
        include_tech = random.random() < prob
        cbt_tech = batch.get('cbt_technique_vi', [None] * bs)[i]

        # Build system prompt
        if cbt_tech and isinstance(cbt_tech, str):
            # Clean technique name (remove quotes if present)
            cbt_tech_clean = cbt_tech[2:-2] if cbt_tech.startswith("['") else cbt_tech
            system_content = get_system_prompt(cbt_tech_clean, include_tech)
        else:
            system_content = get_system_prompt(None, False)
            
        convo = f"{IM_START}{SYSTEM}\n{system_content}{IM_END}\n"

        # Split dialogue into turns
        turns = re.split(r'\n(?=Khách hàng:|Cố vấn:)', dialogue.strip())
        
        for turn in turns:
            turn = turn.strip()
            if turn.startswith("Khách hàng:"):
                text = turn.replace("Khách hàng:", "", 1).strip()
                convo += f"{IM_START}{USER}\n{text}{IM_END}\n"
            elif turn.startswith("Cố vấn:"):
                text = turn.replace("Cố vấn:", "", 1).strip()
                convo += f"{IM_START}{ASSISTANT}\n{text}{IM_END}\n"
            else:
                # Skip any malformed lines
                continue

        out.append(convo)

    return {"text": out}


def load_and_prepare_dataset(csv_path=DATA_PATH, data_size=DATA_SIZE, 
                             test_size=TRAIN_TEST_SPLIT, seed=RANDOM_SEED):
    """
    Load dataset from CSV and prepare train/test splits.
    
    Args:
        csv_path: Path to the CSV file
        data_size: Number of samples to use
        test_size: Fraction of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, test_dataset) as HuggingFace Datasets
    """
    print(f"Loading dataset from {csv_path}...")
    
    # Load and shuffle dataset
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df).shuffle(seed=seed).select(range(data_size))
    
    # Split into train/test
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    train, test = split['train'], split['test']
    
    print(f"Loaded {data_size} samples: {len(train)} train, {len(test)} test")
    
    # Get original columns for removal
    orig_cols = list(train.column_names)
    
    # Format datasets
    print("Formatting train dataset...")
    train_formatted = train.map(
        format_batch_full_chat,
        batched=True,
        remove_columns=orig_cols
    )
    
    print("Formatting test dataset...")
    test_formatted = test.map(
        format_batch_full_chat,
        batched=True,
        remove_columns=orig_cols
    )
    
    print(f"Formatted datasets - Train: {len(train_formatted)}, Test: {len(test_formatted)}")
    
    return train_formatted, test_formatted
