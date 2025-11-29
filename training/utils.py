"""
Utility functions for training and saving models.
"""

import torch

from config import (
    BASE_MODEL_NAME, MODEL_SAVE_NAME, HF_REPO_PREFIX, TRAINING_STEPS
)


def print_gpu_stats(label="Current"):
    """
    Print GPU memory statistics.
    
    Args:
        label: Label for the stats (e.g., "Initial", "After training")
    """
    if not torch.cuda.is_available():
        print(f"{label} GPU stats: CUDA not available")
        return
    
    gpu_stats = torch.cuda.get_device_properties(0)
    memory_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    print(f"{label} GPU stats:")
    print(f"  - Device: {gpu_stats.name}")
    print(f"  - Total memory: {max_memory} GB")
    print(f"  - Reserved memory: {memory_reserved} GB")
    print(f"  - Utilization: {(memory_reserved/max_memory)*100:.1f}%")


def save_model(model, tokenizer, save_name=None):
    """
    Save model and tokenizer locally.
    
    Args:
        model: The trained model
        tokenizer: Model tokenizer
        save_name: Custom save name (optional)
    """
    if save_name is None:
        save_name = MODEL_SAVE_NAME
    
    print(f"Saving model locally to: {save_name}")
    
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    
    print("✓ Model saved successfully")


def push_to_hub(model, tokenizer, hf_token, repo_name=None):
    """
    Push model and tokenizer to HuggingFace Hub.
    
    Args:
        model: The trained model
        tokenizer: Model tokenizer
        hf_token: HuggingFace authentication token
        repo_name: Custom repository name (optional)
    """
    if repo_name is None:
        repo_name = f"{HF_REPO_PREFIX}-{BASE_MODEL_NAME}-{TRAINING_STEPS}"
    
    print(f"Pushing to HuggingFace Hub: {repo_name}")
    
    try:
        model.push_to_hub(repo_name, token=hf_token)
        tokenizer.push_to_hub(repo_name, token=hf_token)
        print(f"✓ Successfully pushed to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"✗ Failed to push to hub: {e}")


def format_training_summary(trainer_stats):
    """
    Format training statistics into a readable summary.
    
    Args:
        trainer_stats: Training statistics from trainer
        
    Returns:
        str: Formatted summary
    """
    summary = f"""
Training Summary:
-----------------
Total steps:     {trainer_stats.global_step}
Training loss:   {trainer_stats.training_loss:.4f}
Best step:       {trainer_stats.best_model_checkpoint or 'N/A'}
"""
    return summary


def validate_environment():
    """
    Validate the training environment.
    
    Returns:
        tuple: (is_valid, list of warnings/errors)
    """
    warnings = []
    
    # Check CUDA
    if not torch.cuda.is_available():
        warnings.append("CUDA is not available - training will be very slow on CPU")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_stats.total_memory / 1024 / 1024 / 1024
        if total_memory_gb < 8:
            warnings.append(f"GPU has only {total_memory_gb:.1f}GB - may not be sufficient for training")
    
    return len(warnings) == 0, warnings
