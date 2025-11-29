"""
Model loading and LoRA configuration for fine-tuning.
"""

from unsloth import FastLanguageModel

from config import (
    BASE_MODEL_NAME, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    USE_GRADIENT_CHECKPOINTING, USE_RSLORA, LOFTQ_CONFIG,
    RANDOM_SEED
)


def load_base_model():
    """
    Load the base pre-trained model with 4-bit quantization.
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {BASE_MODEL_NAME}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{BASE_MODEL_NAME}",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    print("Base model loaded successfully")
    return model, tokenizer


def setup_lora(model):
    """
    Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    
    Args:
        model: The base model to add LoRA adapters to
        
    Returns:
        model: Model with LoRA adapters configured
    """
    print("Setting up LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=RANDOM_SEED,
        use_rslora=USE_RSLORA,
        loftq_config=LOFTQ_CONFIG,
    )
    
    print(f"LoRA configured - Rank: {LORA_R}, Alpha: {LORA_ALPHA}")
    print(f"Target modules: {TARGET_MODULES}")
    
    return model


def prepare_model_for_training():
    """
    Complete model preparation: load base model and configure LoRA.
    
    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer = load_base_model()
    model = setup_lora(model)
    
    return model, tokenizer


def prepare_model_for_inference(model):
    """
    Prepare model for inference (disable training mode optimizations).
    
    Args:
        model: The trained model
        
    Returns:
        model: Model prepared for inference
    """
    print("Preparing model for inference...")
    FastLanguageModel.for_inference(model)
    return model
