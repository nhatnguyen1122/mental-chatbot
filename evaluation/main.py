"""
Main execution script for CBT counselor evaluation.

Usage:
    python main.py [--csv-path PATH] [--output OUTPUT] [--api-key KEY]

Example:
    python main.py --csv-path ../dataset/MentalHealthDataset.csv
"""

import argparse
import os

from models import CounselorModel, GeminiModel
from data_loader import load_data
from conversation import run_conversation_evaluation
from evaluator import aggregate_results, print_aggregate_results
from utils import save_results_to_json, validate_environment, print_welcome_banner
from config import NUM_CONVERSATIONS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate CBT counselor model using simulated conversations'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Path to the Vietnamese CACTUS dataset CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON filename (default: auto-generated)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google API key for Gemini (default: uses GOOGLE_API_KEY env var)'
    )
    parser.add_argument(
        '--num-conversations',
        type=int,
        default=None,
        help=f'Number of conversations to evaluate (default: {NUM_CONVERSATIONS})'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Print welcome banner
    print_welcome_banner()
    
    # Validate environment
    is_valid, errors = validate_environment()
    if not is_valid and args.api_key is None:
        print("⚠️ Environment validation warnings:")
        for error in errors:
            print(f"  - {error}")
        print("\nYou can provide the API key via --api-key argument or set GOOGLE_API_KEY environment variable.")
        print("Continuing anyway...\n")
    
    # Set API key if provided
    if args.api_key:
        os.environ['GOOGLE_API_KEY'] = args.api_key
    
    # Update NUM_CONVERSATIONS if specified
    if args.num_conversations:
        import config
        config.NUM_CONVERSATIONS = args.num_conversations
        print(f"Using {args.num_conversations} conversations\n")
    
    # Load models
    print("=" * 70)
    print("INITIALIZING MODELS")
    print("=" * 70)
    
    counselor_model = CounselorModel()
    if not counselor_model.load():
        print("Failed to load counselor model. Exiting.")
        return 1
    
    gemini_model = GeminiModel()
    if not gemini_model.load():
        print("Failed to load Gemini model. Exiting.")
        return 1
    
    print()
    
    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    initial_prompts, client_backgrounds, initial_attitudes = load_data(args.csv_path)
    
    if not initial_prompts:
        print("No conversation prompts available. Exiting.")
        return 1
    
    print(f"\nReady to evaluate {len(initial_prompts)} conversations.\n")
    
    # Run evaluations
    print("=" * 70)
    print("RUNNING EVALUATIONS")
    print("=" * 70)
    
    conversation_results = []
    
    num_to_evaluate = min(len(initial_prompts), args.num_conversations or NUM_CONVERSATIONS)
    
    for i in range(num_to_evaluate):
        result = run_conversation_evaluation(
            initial_prompts[i],
            client_backgrounds[i],
            initial_attitudes[i],
            counselor_model,
            gemini_model
        )
        conversation_results.append(result)
        print(f"\nĐã hoàn thành {i+1}/{num_to_evaluate} hội thoại.\n{'='*70}\n")
    
    # Aggregate and display results
    print("=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)
    
    criteria_stats, total_average = aggregate_results(conversation_results)
    num_evaluated = len([r for r in conversation_results if r.get('scores')])
    
    print_aggregate_results(criteria_stats, total_average, num_evaluated)
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    save_results_to_json(conversation_results, criteria_stats, total_average, args.output)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
