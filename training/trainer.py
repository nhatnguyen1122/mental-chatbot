"""
Training loop and configuration using SFTTrainer.
"""

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from config import (
    PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    TRAINING_STEPS, LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY,
    LR_SCHEDULER_TYPE, OPTIMIZER, RANDOM_SEED, OUTPUT_DIR,
    LOGGING_STEPS, REPORT_TO, MAX_SEQ_LENGTH, DATASET_NUM_PROC, PACKING
)


def create_trainer(model, tokenizer, train_dataset):
    """
    Create and configure the SFTTrainer for fine-tuning.
    
    Args:
        model: The model with LoRA adapters
        tokenizer: Model tokenizer
        train_dataset: Formatted training dataset
        
    Returns:
        SFTTrainer: Configured trainer
    """
    print("Creating trainer...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=DATASET_NUM_PROC,
        packing=PACKING,
        args=TrainingArguments(
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_steps=TRAINING_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=LOGGING_STEPS,
            optim=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=RANDOM_SEED,
            output_dir=OUTPUT_DIR,
            report_to=REPORT_TO,
        ),
    )
    
    print(f"Trainer configured:")
    print(f"  - Training steps: {TRAINING_STEPS}")
    print(f"  - Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Optimizer: {OPTIMIZER}")
    
    return trainer


def train_model(trainer):
    """
    Execute the training loop.
    
    Args:
        trainer: Configured SFTTrainer
        
    Returns:
        TrainerState: Training statistics
    """
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    trainer_stats = trainer.train()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    return trainer_stats
