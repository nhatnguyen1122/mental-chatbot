"""
Configuration settings for CBT evaluation framework.
"""

# Model Configuration
MODEL_NAME = 'CACTUS-Qwen3-4B-300'
FINETUNED_MODEL_PATH = f"PQPQPQHUST/{MODEL_NAME}"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Model Parameters
MAX_SEQ_LENGTH = 32768
DTYPE = None  # Auto derived from model. For Qwen2.5, it's bfloat16.
LOAD_IN_4BIT = True

# Conversation Configuration
NUM_CONVERSATIONS = 100
NUMBER_OF_TURNS_PER_CONVERSATION = 30

# Role Labels (Vietnamese)
SYSTEM = "Hệ thống"
USER = "Người dùng"
ASSISTANT = "Cố vấn"
END_SESSION_TOKEN = "[END_CONVERSATION]"

# Dataset Configuration
DIALOGUE_COLUMN_NAME = 'dialogue_vi'
INTAKE_FORM_COLUMN_NAME = 'intake_form_vi'
ATTITUDE_COLUMN_NAME = 'attitude_vi'

# API Timeouts
GEMINI_CLIENT_TIMEOUT = 30  # seconds
GEMINI_EVAL_TIMEOUT = 60  # seconds

# Generation Parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.95
MIN_P = 0.05

# Evaluation Criteria
CRITERIA = {
    "understanding": "Sự thấu hiểu",
    "interpersonal_effectiveness": "Hiệu quả tương tác",
    "collaboration": "Sự hợp tác",
    "guided_discovery": "Khám phá có hướng dẫn",
    "focusing": "Tập trung vào các Nhận thức hoặc Hành vi chính",
    "strategy_for_change": "Chiến lược thay đổi",
    "attitude_change": "Thay đổi Thái độ tiếp nhận giải pháp"
}

# Attitude Descriptions
ATTITUDES = {
    'Tiêu cực': 'Cho thấy thái độ bi quan, lo lắng, buồn bã trước vấn đề tâm lí, cần nhiều thời gian hơn để được thuyết phục chấp nhận lời khuyên hoặc thực hiện giải pháp',
    'Trung lập': 'Thể hiện thái độ trung lập, bình thường trước các giải pháp tâm lí, sẽ cân nhắc thực hiện, thay đổi nếu các lời khuyên, giải pháp là hợp lí và thuyết phục',
    'Tích cực': 'Sẵn lòng tiếp nhận các lời khuyên, câu hỏi và giải pháp nếu điều đó hợp lí và giúp ích cho vấn đề hiện tại'
}

# Default Data (fallback when dataset is insufficient)
DEFAULT_DATA_SETS = [
    (
        "Tôi thường xuyên cảm thấy lo lắng về công việc và không biết nên bắt đầu từ đâu.",
        'Tiêu cực',
        "Khách hàng là nam giới, 30 tuổi, làm IT. Gần đây gặp nhiều áp lực deadline, cảm thấy bất an, khó ngủ. Chưa từng đi tư vấn tâm lí bao giờ."
    ),
    (
        "Tôi thấy khó khăn trong việc đối phó với những suy nghĩ tiêu cực.",
        'Trung lập',
        "Khách hàng là nữ, 25 tuổi, sinh viên. Thường có suy nghĩ tự ti về bản thân, so sánh với người khác trên mạng xã hội. Đã đọc sách về tâm lí nhưng chưa cải thiện."
    ),
]
