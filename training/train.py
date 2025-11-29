
from unsloth import FastLanguageModel
import torch
import re
import random
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import sys
sys.path.append("..")

model_name="Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
step=100
data_size=6000
HF_TOKEN=''
max_seq_length=2048

lr = 1e-4
per_device_train_batch_size = 4
gradient_accumulation_steps = 4

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"unsloth/{model_name}",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    # token = "", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimzed
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    #modules_to_save=["lm_head", "embed_tokens"]
)

IM_START  = "<|im_start|>"
IM_END    = "<|im_end|>"
SYSTEM    = "HỆ THỐNG"
USER      = "NGƯỜI DÙNG"
ASSISTANT = "CỐ VẤN"

# List of CBT techniques and descriptions
cbt_techniques = {
    "Giảm thiểu thảm họa hoá": 
        "Giúp người dùng giảm bớt nỗi sợ kịch bản tồi tệ nhất.",
    "Góc nhìn thay thế": 
        "Khuyến khích xem xét tình huống từ một góc độ khác.",
    "Đặt câu hỏi dựa trên bằng chứng": 
        "Hướng dẫn người dùng tìm bằng chứng ủng hộ hoặc bác bỏ niềm tin của họ.",
    "Thí nghiệm hành vi": 
        "Gợi ý người dùng thực hiện thử nghiệm hành vi để kiểm chứng niềm tin trong thực tế.",
    "Kiểm tra thực tế": 
        "Giúp người dùng so sánh niềm tin của họ với bằng chứng thực tế.",
    "Đánh giá hiệu quả": 
        "Đánh giá xem mô hình suy nghĩ hoặc hành vi của người dùng có thực tế và có ích hay không.",
    "Chuyển đổi quy tắc Thành mong muốn": 
        "Khuyến khích chuyển các câu 'cần phải/đáng lẽ' cứng nhắc thành mong muốn linh hoạt.",
    "Huấn luyện kỹ năng giải quyết vấn đề": 
        "Hướng dẫn người dùng phương pháp có cấu trúc để xác định và giải quyết vấn đề."
}

def get_system_prompt(cbt_technique=None, include_technique=False):
    base_prompt = (
        "Bạn là một trợ lý tư vấn chuyên nghiệp, thấu hiểu và dựa trên Liệu pháp Hành vi Nhận thức (CBT). "
        "Hãy phân tích tình huống của khách hàng và phản hồi phù hợp để hỗ trợ sức khỏe tinh thần của họ."
    )

    if include_technique and cbt_technique:
        desc = cbt_techniques.get(cbt_technique, "")
        return (
            f"{base_prompt} Hãy tập trung áp dụng kỹ thuật '{cbt_technique}': {desc}"
            if desc else f"{base_prompt} Hãy áp dụng kỹ thuật '{cbt_technique}'."
        )
    else:
        techniques_list = "\n".join([
            f"{i+1}. {name} – {desc}"
            for i, (name, desc) in enumerate(cbt_techniques.items())
        ])
        return (
            f"{base_prompt} Dựa vào vấn đề của khách hàng, hãy lựa chọn và áp dụng kỹ thuật CBT "
            f"phù hợp nhất từ danh sách sau:\n\n{techniques_list}"
        )

def format_batch_full_chat(batch, prob=0.5):
    """
    For each example in the batch, emit exactly one training string:
      [ system prompt ]
      <|im_start|>user   <client turn 1>
      <|im_end|>
      <|im_start|>assistant <Cố vấn turn 1>
      <|im_end|>
      ...
    """
    out = []
    bs = len(batch['dialogue_vi'])
    for i in range(bs):
        dialogue = batch['dialogue_vi'][i]
        if not dialogue:
            continue

        # decide whether to mention a specific technique
        include_tech = random.random() < prob
        cbt_tech = batch.get('cbt_technique_vi', [None]*bs)[i]

        # build system prompt
        system_content = get_system_prompt(cbt_tech[2:-2], include_tech)
        convo = f"{IM_START}{SYSTEM}\n{system_content}{IM_END}\n"

        # split into turns
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
                # skip any malformed lines
                continue

        out.append(convo)

    return {"text": out}

dataset = Dataset.from_pandas(pd.read_csv("dataset/MentalHealthDataset.csv")).shuffle(seed=42).select(range(data_size))
split = dataset.train_test_split(test_size=0.2 , seed=42)
train, test = split['train'], split['test']

# assume `train` and `test` are already loaded HuggingFace Datasets
orig_cols = list(train.column_names)

cactus_train = train.map(
    format_batch_full_chat,
    batched=True,
    remove_columns=orig_cols
)

cactus_test = test.map(
    format_batch_full_chat,
    batched=True,
    remove_columns=orig_cols
)

print(f"Train examples: {len(cactus_train)}, Test examples: {len(cactus_test)}")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = cactus_train,
    dataset_text_field = "text",
    max_seq_length = 2048,
    #formatting_func=format_batch_full_chat,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 50,
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = step,
        learning_rate = lr,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()

FastLanguageModel.for_inference(model)
prompt = """
<|im_start|>system
Bạn là một trợ lý tư vấn chuyên nghiệp, thấu hiểu và dựa trên Liệu pháp Hành vi Nhận thức (CBT). Hãy phân tích tình huống của khách hàng và phản hồi phù hợp để hỗ trợ sức khỏe tinh thần của họ.<|im_end|>
<|im_start|>user
Gần đây tôi cảm thấy mình bị mắc kẹt trong vòng luẩn quẩn của việc suy nghĩ quá nhiều về mọi thứ. Ngay cả những quyết định nhỏ nhất—như chọn ăn gì hay gửi tin nhắn gì—cũng trở nên mệt mỏi. Tôi cứ tua đi tua lại các cuộc trò chuyện trong đầu, tự hỏi liệu mình có nói gì sai không, hay liệu mọi người có ngầm ghét tôi không. Tôi biết điều đó là phi lý, nhưng tôi không thể dừng lại được. Nó bắt đầu ảnh hưởng đến giấc ngủ và đời sống xã hội của tôi. Bạn có nghĩ tôi có vấn đề gì không? Hay tôi chỉ đang làm quá lên?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens = 300, use_cache = True)
tokenizer.batch_decode(outputs)

model.save_pretrained(f"{model_name}") # Local saving
tokenizer.save_pretrained(f"{model_name}")
model.push_to_hub(f"PQPQPQHUST/CACTUS-{model_name}-{step}", token = HF_TOKEN) # Online saving
tokenizer.push_to_hub(f"PQPQPQHUST/CACTUS-{model_name}-{step}", token = HF_TOKEN) # Online saving