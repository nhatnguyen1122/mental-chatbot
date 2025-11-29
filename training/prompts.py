"""
CBT techniques and system prompt generation for Vietnamese counseling.
"""

# Vietnamese CBT Techniques
CBT_TECHNIQUES = {
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
    """
    Generate system prompt for the counselor model.
    
    Args:
        cbt_technique: Specific CBT technique to focus on (optional)
        include_technique: Whether to mention the specific technique in the prompt
        
    Returns:
        str: System prompt text
    """
    base_prompt = (
        "Bạn là một trợ lý tư vấn chuyên nghiệp, thấu hiểu và dựa trên Liệu pháp Hành vi Nhận thức (CBT). "
        "Hãy phân tích tình huống của khách hàng và phản hồi phù hợp để hỗ trợ sức khỏe tinh thần của họ."
    )

    if include_technique and cbt_technique:
        desc = CBT_TECHNIQUES.get(cbt_technique, "")
        return (
            f"{base_prompt} Hãy tập trung áp dụng kỹ thuật '{cbt_technique}': {desc}"
            if desc else f"{base_prompt} Hãy áp dụng kỹ thuật '{cbt_technique}'."
        )
    else:
        techniques_list = "\n".join([
            f"{i+1}. {name} – {desc}"
            for i, (name, desc) in enumerate(CBT_TECHNIQUES.items())
        ])
        return (
            f"{base_prompt} Dựa vào vấn đề của khách hàng, hãy lựa chọn và áp dụng kỹ thuật CBT "
            f"phù hợp nhất từ danh sách sau:\n\n{techniques_list}"
        )


# Test prompts for inference
TEST_PROMPT_TEMPLATE = """<|im_start|>system
Bạn là một trợ lý tư vấn chuyên nghiệp, thấu hiểu và dựa trên Liệu pháp Hành vi Nhận thức (CBT). Hãy phân tích tình huống của khách hàng và phản hồi phù hợp để hỗ trợ sức khỏe tinh thần của họ.<|im_end|>
<|im_start|>user
Gần đây tôi cảm thấy mình bị mắc kẹt trong vòng luẩn quẩn của việc suy nghĩ quá nhiều về mọi thứ. Ngay cả những quyết định nhỏ nhất—như chọn ăn gì hay gửi tin nhắn gì—cũng trở nên mệt mỏi. Tôi cứ tua đi tua lại các cuộc trò chuyện trong đầu, tự hỏi liệu mình có nói gì sai không, hay liệu mọi người có ngầm ghét tôi không. Tôi biết điều đó là phi lý, nhưng tôi không thể dừng lại được. Nó bắt đầu ảnh hưởng đến giấc ngủ và đời sống xã hội của tôi. Bạn có nghĩ tôi có vấn đề gì không? Hay tôi chỉ đang làm quá lên?<|im_end|>
<|im_start|>assistant
"""
