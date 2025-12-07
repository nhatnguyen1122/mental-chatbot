# Towards a Vietnamese Mental Health Support Chatbot with Large Language Models

A comprehensive framework for training and evaluating Vietnamese mental health chatbots using Cognitive Behavioral Therapy (CBT) techniques. This paper includes both a fine-tuning pipeline for counselor models and an automated evaluation system using simulated conversations.

## 🌟 Features

### Training Framework
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
- **4-bit Quantization**: Memory-efficient training with QLoRA
- **Unsloth Optimization**: VRAM reduction for faster training
- **CBT Integration**: Built-in Vietnamese CBT techniques and prompting
- **Interactive Testing**: Test your model during and after training
- **Flexible Saving**: Save to local disk or push directly to HuggingFace Hub

### Evaluation Framework
- **Automated Evaluation**: Run multiple simulated conversations automatically
- **CTRS-Based Scoring**: 7-criteria evaluation based on Cognitive Therapy Rating Scale
- **Gemini-Powered**: Uses Gemini models for client simulation and judging
- **Comprehensive Metrics**: Detailed scoring across multiple therapeutic dimensions
- **JSON Export**: Results saved in structured format for analysis

## 📁 Project Structure

```
mental-chatbot/
├── training/                      # Model fine-tuning pipeline
│   ├── config.py                 # Training hyperparameters
│   ├── prompts.py                # CBT techniques and system prompts
│   ├── data_formatter.py         # Dataset loading and formatting
│   ├── model_setup.py            # Model loading and LoRA setup
│   ├── trainer.py                # Training loop with SFTTrainer
│   ├── inference.py              # Model testing and inference
│   ├── utils.py                  # GPU stats, saving, utilities
│   ├── main.py                   # Training entry point
│   ├── train.py                  # Original script (preserved)
│   ├── README.md                 # Detailed training docs
│   └── requirements.txt          # Training dependencies
│
├── evaluation/                    # Evaluation framework
│   ├── config.py                 # Evaluation settings
│   ├── prompts.py                # Prompts and CBT techniques
│   ├── models.py                 # Model wrappers
│   ├── evaluator.py              # Scoring logic
│   ├── conversation.py           # Conversation management
│   ├── data_loader.py            # CSV data loading
│   ├── utils.py                  # Helper functions
│   ├── main.py                   # Evaluation entry point
│   ├── eval-qwen-no-reasoning.ipynb  # Original notebook
│   ├── README.md                 # Detailed evaluation docs
│   └── requirements.txt          # Evaluation dependencies
│
├── dataset/                       # Training and evaluation data
│   └── MentalHealthDataset.csv   # Vietnamese mental health conversations
│
├── README.md                      # This file
└── requirements.txt               # Unified dependencies
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
cd /path/to/mental-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# For evaluation (Gemini API)
export GOOGLE_API_KEY="your-gemini-api-key"

# For training (optional - only if pushing to HuggingFace)
export HF_TOKEN="your-huggingface-token"
```

### Training a Model

**Basic training:**
```bash
cd training
python main.py
```

**Interactive testing mode:**
```bash
python main.py --interactive
```

**Test existing model without training:**
```bash
python main.py --test-only
```

**Skip saving (for quick experiments):**
```bash
python main.py --skip-save
```

**Push to HuggingFace Hub:**
```bash
python main.py --hf-token YOUR_TOKEN
```

### Evaluating a Model

**Run full evaluation:**
```bash
cd evaluation
python main.py --csv-path ../dataset/MentalHealthDataset.csv --output results.json --api-key YOUR_GEMINI_KEY
```

**Evaluate specific number of conversations:**
```bash
python main.py --csv-path ../dataset/MentalHealthDataset.csv --num-conversations 50
```

**Custom output location:**
```bash
python main.py --csv-path ../dataset/MentalHealthDataset.csv --output ../results/eval_$(date +%Y%m%d).json
```

## 🧠 CBT Techniques

Both frameworks use 8 Vietnamese Cognitive Behavioral Therapy techniques:

1. **Decatastrophizing (Giảm thiểu thảm họa hoá)**
   - Aims to reduce the habit of imagining the worst-case scenario by assessing how likely the feared outcome really is and preparing ways to cope.

2. **Alternative Perspective (Góc nhìn thay thế)**
   - Encourages clients to consider how others might view similar situations, promoting openness to different interpretations.

3. **Evidence-Based Questioning (Đặt câu hỏi dựa trên bằng chứng)**
   - Guides clients to examine evidence that supports or contradicts their thoughts, fostering a more fact-based way of thinking.

4. **Behavior Experiment (Thí nghiệm hành vi)**
   - Involves testing new behaviors in specific situations to challenge and change negative beliefs.

5. **Reality Testing (Kiểm tra thực tế)**
   - Helps clients evaluate how accurately their thoughts reflect reality, distinguishing between perceptions and actual experiences.

6. **Efficiency Evaluation (Đánh giá hiệu quả)**
   - Assists individuals in assessing how useful their thoughts or beliefs are, analyzing their practicality or potential harm in real-life contexts.

7. **Changing Rules to Wishes (Chuyển đổi quy tắc thành mong muốn)**
   - Replaces rigid rules or unrealistic standards with achievable hopes or desires.

8. **Problem-Solving Skills Training (Huấn luyện kỹ năng giải quyết vấn đề)**
   - Teaches systematic approaches to tackling problems, including identifying issues, generating possible solutions, and implementing them.

## 📊 Evaluation Metrics

The evaluation framework uses 7 criteria based on Cognitive Therapy Rating Scale (CTRS) and custom metrics, each scored on a 0-6 scale:

### General Counseling Skills

| Criterion | Description |
|-----------|-------------|
| **Understanding** | Therapist's ability to accurately comprehend the client's issues, emotions, and underlying concerns |
| **Interpersonal Effectiveness** | Capacity to foster a positive, supportive, and trusting relationship (warmth, genuineness, rapport) |
| **Collaboration** | How well the therapist involves the client in the therapeutic process (joint goal-setting, active participation) |

### CBT-Specific Skills

| Criterion | Description |
|-----------|-------------|
| **Guided Discovery** | Use of Socratic questioning to guide the client toward self-reflection and insight |
| **Focus** | Ability to identify and prioritize key cognitions or behaviors, maintaining clear focus throughout the session |
| **Strategy** | Coherence and appropriateness of the overall approach, including evidence-based CBT techniques |

### Custom Metric

| Criterion | Description |
|-----------|-------------|
| **Change in Attitude Towards Guidance** | Observed change in client's attitude regarding counselor's guidance throughout the session (from negative/neutral/positive initial state) |


## 🔧 Configuration

### Training Configuration (`training/config.py`)

Key parameters you can adjust:

```python
# Model settings
BASE_MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# LoRA settings
LORA_R = 16                    # Rank (higher = more parameters)
LORA_ALPHA = 16                # Scaling factor
LORA_DROPOUT = 0.05            # Regularization

# Training settings
TRAINING_STEPS = 100           # Number of training steps
PER_DEVICE_TRAIN_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 10

# Dataset
DATASET_NAME = "nhat2105/MentalHealthDataset"
```

### Evaluation Configuration (`evaluation/config.py`)

Key parameters you can adjust:

```python
# Evaluation settings
NUM_CONVERSATIONS = 100        # Conversations to evaluate
MAX_TURNS = 6                  # Turns per conversation
TIMEOUT_SECONDS = 30           # Timeout per turn

# Models
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Scoring
CRITERIA = {
    "agenda_setting": ...,
    "feedback": ...,
    # ... 7 total criteria
}
```


## 🖥️ System Requirements

### Minimum Requirements
- **GPU**: 16GB VRAM (NVIDIA recommended)
- **RAM**: 32GB system memory
- **Storage**: 50GB free space
- **Python**: 3.10 or higher

### Recommended Setup
- **GPU**: 24GB VRAM (RTX 3090/4090, A5000, etc.)
- **RAM**: 64GB system memory
- **Storage**: 100GB SSD
- **CUDA**: 12.1 or higher

### Cloud Alternatives
- **Google Colab**: T4 GPU (free tier may work with batch_size=1)
- **Kaggle**: P100 GPU (16GB VRAM)
- **AWS**: g5.xlarge or higher
- **Vast.ai**: RTX 3090 instances (~$0.30/hour)


## 🤝 Integration

### Using the Trained Model

**Load in Python:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="model-unsloth",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Generate response
inputs = tokenizer(["<|im_start|>user\nTôi cảm thấy lo lắng<|im_end|>\n<|im_start|>assistant\n"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Deploy as API:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat(message: Message):
    # Use inference.py logic
    response = generate_response(message.text)
    return {"response": response}
```

## 📚 Additional Resources

### Research Papers
- [CACTUS: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory](https://arxiv.org/abs/2407.03103)

### External Links
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Gemini API Docs](https://ai.google.dev/docs)
- [CBT Overview](https://www.apa.org/ptsd-guideline/patients-and-families/cognitive-behavioral)

## 🔬 Advanced Usage

### Custom Dataset Format

```python
# In training/data_formatter.py, modify load_and_prepare_dataset():
def load_custom_dataset(path):
    data = pd.read_csv(path)
    # Your custom formatting logic
    return formatted_data
```

### Multi-GPU Training

```python
# Set in training/config.py:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Or use accelerate:
accelerate launch --multi_gpu --num_processes=4 main.py
```

### Custom Evaluation Criteria

```python
# In evaluation/config.py, add new criteria:
CRITERIA = {
    "custom_criterion": {
        "name": "My Custom Criterion",
        "description": "Evaluates...",
        "scale": {0: "Poor", ..., 6: "Excellent"}
    }
}

# Update prompts.py to include in evaluation template
```

## 📄 License

This project is intended for research and educational purposes in mental health support systems.

## ⚠️ Disclaimer

This chatbot is a research tool and should not replace professional mental health care. Always encourage users experiencing serious mental health issues to seek help from qualified professionals.

## 🙏 Acknowledgments

- **Unsloth**: For efficient fine-tuning optimizations
- **Qwen Team**: For the excellent base model
- **Google**: For Gemini API access
- **HuggingFace**: For model hosting and transformers library
- **CACTUS Paper**: For the foundational work on the dataset and CBT-based evaluation frameworks

## 📧 Contact

For questions or issues, please contact: **nhat.ntm235986@sis.hust.edu.vn**

---

