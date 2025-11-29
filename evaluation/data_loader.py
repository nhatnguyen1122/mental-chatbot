"""
Data loading utilities for the Vietnamese CACTUS dataset.
"""

import random
import pandas as pd

from config import (
    NUM_CONVERSATIONS, DIALOGUE_COLUMN_NAME, 
    INTAKE_FORM_COLUMN_NAME, ATTITUDE_COLUMN_NAME,
    DEFAULT_DATA_SETS
)


def extract_first_client_message(dialogue):
    """Extract the first client message from a dialogue using Vietnamese label."""
    if not isinstance(dialogue, str):
        return None
    
    # Find the first occurrence of "Khách hàng:"
    client_start = dialogue.find("Khách hàng:")
    if client_start == -1:
        return None
    
    # Skip "Khách hàng:" to get content
    content_start = client_start + len("Khách hàng:")
    
    # Find the next "Cố vấn:"
    next_advisor = dialogue.find("Cố vấn:", content_start)
    if next_advisor == -1:
        # If not found, take until the end of string
        client_message = dialogue[content_start:].strip()
    else:
        client_message = dialogue[content_start:next_advisor].strip()
    
    return client_message


def load_dataset_from_csv(csv_path):
    """
    Load Vietnamese CACTUS dataset from a local CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        tuple: (initial_prompts, client_backgrounds, initial_attitudes)
    """
    initial_prompts = []
    client_backgrounds = []
    initial_attitudes = []
    
    try:
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Clean attitude column
        if ATTITUDE_COLUMN_NAME in df.columns:
            df[ATTITUDE_COLUMN_NAME] = df[ATTITUDE_COLUMN_NAME].apply(lambda x: x[2:-2] if isinstance(x, str) and len(x) > 4 else x)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Loaded {len(df)} rows from dataset")
        
        # Check required columns
        required_columns = [DIALOGUE_COLUMN_NAME, INTAKE_FORM_COLUMN_NAME, ATTITUDE_COLUMN_NAME]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Warning: Missing required columns: {missing}")
            return initial_prompts, client_backgrounds, initial_attitudes
        
        # Filter valid rows
        valid_rows = df.dropna(subset=required_columns).copy()
        
        # Extract first message
        valid_rows['first_message'] = valid_rows[DIALOGUE_COLUMN_NAME].apply(extract_first_client_message)
        
        # Filter rows with valid first message (> 20 chars)
        filtered_rows = valid_rows[valid_rows['first_message'].str.len() > 20].copy()
        
        if not filtered_rows.empty:
            # Sample rows
            if len(filtered_rows) > NUM_CONVERSATIONS:
                sampled_rows = filtered_rows.sample(n=NUM_CONVERSATIONS, random_state=42)
            else:
                sampled_rows = filtered_rows
            
            # Extract data
            for _, row in sampled_rows.iterrows():
                prompt = row['first_message']
                client_background_text = row[INTAKE_FORM_COLUMN_NAME]
                attitude_label = row[ATTITUDE_COLUMN_NAME]
                
                initial_prompts.append(prompt)
                client_backgrounds.append(client_background_text)
                initial_attitudes.append(attitude_label)
            
            print(f"Extracted {len(initial_prompts)} conversation prompts from dataset.")
        else:
            print("No valid conversation prompts found in dataset.")
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    return initial_prompts, client_backgrounds, initial_attitudes


def load_data(csv_path=None):
    """
    Load conversation data from CSV or use defaults.
    
    Args:
        csv_path: Optional path to CSV file. If None, uses defaults only.
        
    Returns:
        tuple: (initial_prompts, client_backgrounds, initial_attitudes)
    """
    initial_prompts = []
    client_backgrounds = []
    initial_attitudes = []
    
    # Try to load from CSV if path provided
    if csv_path:
        initial_prompts, client_backgrounds, initial_attitudes = load_dataset_from_csv(csv_path)
    
    # Add defaults if needed
    if len(initial_prompts) < NUM_CONVERSATIONS:
        print(f"Dataset has {len(initial_prompts)} prompts, need {NUM_CONVERSATIONS}. Adding defaults...")
        
        default_sets = list(DEFAULT_DATA_SETS)
        random.seed(42)
        
        while len(initial_prompts) < NUM_CONVERSATIONS and default_sets:
            default_item = default_sets.pop(0)
            if len(default_item) == 3:
                initial_prompts.append(default_item[0])
                initial_attitudes.append(default_item[1])
                client_backgrounds.append(default_item[2])
            else:
                print(f"Warning: Skipping incomplete default data point: {default_item}")
        
        if len(initial_prompts) < NUM_CONVERSATIONS:
            print(f"Warning: Only {len(initial_prompts)} prompts available (requested {NUM_CONVERSATIONS})")
    
    return initial_prompts, client_backgrounds, initial_attitudes
