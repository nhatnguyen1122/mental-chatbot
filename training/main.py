"""
Main training script for Vietnamese CBT counselor model.

Usage:
    python main.py [--test-only] [--interactive]
"""

import argparse
import os

from model_setup import prepare_model_for_training, prepare_model_for_inference
from data_formatter import load_and_prepare_dataset
from trainer import create_trainer, train_model
from inference import test_model, interactive_test
from utils import print_gpu_stats, save_model, push_to_hub
from config import HF_TOKEN


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune Vietnamese CBT counselor model'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Skip training and only test the model'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive chat mode after training/testing'
    )
    parser.add_argument(
        '--skip-save',
        action='store_true',
        help='Skip saving the model locally and to HuggingFace'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace token for pushing model (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Print banner
    print("\n" + "="*70)
    print("VIETNAMESE CBT COUNSELOR - FINE-TUNING")
    print("="*70 + "\n")
    
    # Show initial GPU stats
    print_gpu_stats("Initial")
    
    # Prepare model
    print("\n" + "="*70)
    print("MODEL PREPARATION")
    print("="*70 + "\n")
    
    model, tokenizer = prepare_model_for_training()
    
    if not args.test_only:
        # Load and prepare dataset
        print("\n" + "="*70)
        print("DATA PREPARATION")
        print("="*70 + "\n")
        
        train_dataset, test_dataset = load_and_prepare_dataset()
        
        # Create trainer
        trainer = create_trainer(model, tokenizer, train_dataset)
        
        # Train
        trainer_stats = train_model(trainer)
        
        print(f"\nTraining statistics:")
        print(f"  - Total steps: {trainer_stats.global_step}")
        print(f"  - Training loss: {trainer_stats.training_loss:.4f}")
    
    # Prepare for inference
    model = prepare_model_for_inference(model)
    
    # Test the model
    print("\n" + "="*70)
    print("MODEL TESTING")
    print("="*70 + "\n")
    
    test_model(model, tokenizer)
    
    # Interactive mode
    if args.interactive:
        interactive_test(model, tokenizer)
    
    # Save model
    if not args.skip_save:
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70 + "\n")
        
        # Save locally
        save_model(model, tokenizer)
        
        # Push to HuggingFace Hub
        hf_token = args.hf_token or HF_TOKEN or os.environ.get("HF_TOKEN")
        if hf_token:
            push_to_hub(model, tokenizer, hf_token)
        else:
            print("⚠️ No HuggingFace token provided. Skipping hub upload.")
            print("   Set HF_TOKEN in config.py or use --hf-token argument")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
