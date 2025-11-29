"""
Training configuration for Vietnamese CBT counselor model.
"""

# Model Configuration
BASE_MODEL_NAME = "Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect
LOAD_IN_4BIT = True

# LoRA Configuration
LORA_R = 16  # LoRA rank
LORA_ALPHA = 16
LORA_DROPOUT = 0  # 0 is optimized
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
USE_GRADIENT_CHECKPOINTING = "unsloth"  # "unsloth" uses 30% less VRAM
USE_RSLORA = False
LOFTQ_CONFIG = None

# Training Hyperparameters
TRAINING_STEPS = 100
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
OPTIMIZER = "adamw_8bit"
RANDOM_SEED = 3407

# Data Configuration
DATA_PATH = "dataset/MentalHealthDataset.csv"
DATA_SIZE = 6000
TRAIN_TEST_SPLIT = 0.2  # 20% for test
TECHNIQUE_MENTION_PROB = 0.5  # Probability of mentioning specific CBT technique

# Output Configuration
OUTPUT_DIR = "outputs"
MODEL_SAVE_NAME = BASE_MODEL_NAME  # Local save name
HF_REPO_PREFIX = "PQPQPQHUST/CACTUS"  # HuggingFace repository prefix

# HuggingFace Token (set via environment variable or here)
HF_TOKEN = ''  # Leave empty to use environment variable

# Chat Format Tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Role Labels (Vietnamese)
SYSTEM = "HỆ THỐNG"
USER = "NGƯỜI DÙNG"
ASSISTANT = "CỐ VẤN"

# Dataset Processing
DATASET_NUM_PROC = 2
PACKING = False  # Set True for short sequences (5x faster)

# Logging
LOGGING_STEPS = 1
REPORT_TO = "none"  # Options: "wandb", "tensorboard", "none"
