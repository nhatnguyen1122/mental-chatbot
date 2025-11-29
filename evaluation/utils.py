"""
Utility functions for the evaluation framework.
"""

import json
import numpy as np

from config import MODEL_NAME, GEMINI_MODEL_NAME, NUM_CONVERSATIONS, NUMBER_OF_TURNS_PER_CONVERSATION, CRITERIA


def save_results_to_json(conversation_results, criteria_stats, total_average, 
                         output_filename=None):
    """
    Save comprehensive evaluation results to JSON file.
    
    Args:
        conversation_results: List of conversation evaluation results
        criteria_stats: Statistics for each criterion
        total_average: Overall average score
        output_filename: Optional custom output filename
    """
    if output_filename is None:
        output_filename = f'CACTUS_eval_{MODEL_NAME}_{NUM_CONVERSATIONS}.json'
    
    try:
        # Ensure data is serializable
        serializable_criteria_stats = {}
        for crit, stats in criteria_stats.items():
            serializable_criteria_stats[crit] = {
                k: (float(v) if isinstance(v, (np.generic, np.ndarray)) else v) 
                for k, v in stats.items()
            }
        
        comprehensive_results = {
            "config": {
                "finetuned_model": f"PQPQPQHUST/{MODEL_NAME}",
                "gemini_model": GEMINI_MODEL_NAME,
                "num_conversations": NUM_CONVERSATIONS,
                "num_turns_per_conversation": NUMBER_OF_TURNS_PER_CONVERSATION,
                "evaluation_criteria": CRITERIA,
                "model_format": "Qwen Instruct"
            },
            "conversations": [
                {
                    "initial_prompt": result["initial_prompt"],
                    "attitude": result.get("attitude", "Unknown"),
                    "num_turns_executed": len(result["bot_turns"]),
                    "evaluation_scores": result.get("scores"),
                    "evaluation_raw_text": result.get("evaluation_text")
                } for result in conversation_results
            ],
            "criteria_statistics": serializable_criteria_stats,
            "total_average": float(total_average) if isinstance(total_average, (np.generic, np.ndarray)) else total_average
        }
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nKết quả đã được lưu vào file '{output_filename}'")
        return True
    
    except Exception as e:
        print(f"Lỗi khi lưu kết quả: {e}")
        return False


def validate_environment():
    """
    Validate that the environment is properly set up.
    Returns tuple (is_valid, error_messages)
    """
    errors = []
    
    # Check for API key
    import os
    if not os.environ.get("GOOGLE_API_KEY"):
        errors.append("GOOGLE_API_KEY environment variable not set")
    
    return len(errors) == 0, errors


def print_welcome_banner():
    """Print welcome banner with configuration info."""
    print("=" * 70)
    print("CBT Counselor Evaluation Framework")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Evaluator: {GEMINI_MODEL_NAME}")
    print(f"Conversations: {NUM_CONVERSATIONS}")
    print(f"Turns per conversation: {NUMBER_OF_TURNS_PER_CONVERSATION}")
    print("=" * 70)
    print()
