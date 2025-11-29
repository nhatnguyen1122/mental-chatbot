"""
Model inference and testing utilities.
"""

from prompts import TEST_PROMPT_TEMPLATE


def test_model(model, tokenizer, prompt=None, max_new_tokens=300):
    """
    Test the trained model with a sample prompt.
    
    Args:
        model: The trained model (should be in inference mode)
        tokenizer: Model tokenizer
        prompt: Custom prompt (optional, uses default test prompt if None)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated response
    """
    if prompt is None:
        prompt = TEST_PROMPT_TEMPLATE
    
    print("\n" + "="*70)
    print("TESTING MODEL")
    print("="*70)
    print("\nPrompt:")
    print(prompt)
    print("\n" + "-"*70)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        use_cache=True
    )
    
    # Decode and display
    decoded = tokenizer.batch_decode(outputs)
    
    print("\nGenerated response:")
    print(decoded[0])
    print("="*70 + "\n")
    
    return decoded[0]


def interactive_test(model, tokenizer):
    """
    Interactive testing mode - chat with the model.
    
    Args:
        model: The trained model (should be in inference mode)
        tokenizer: Model tokenizer
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Type 'quit' or 'exit' to stop")
    print("="*70 + "\n")
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break
        
        if not user_input:
            continue
        
        # Build prompt
        prompt = f"""<|im_start|>system
Bạn là một trợ lý tư vấn chuyên nghiệp, thấu hiểu và dựa trên Liệu pháp Hành vi Nhận thức (CBT). Hãy phân tích tình huống của khách hàng và phản hồi phù hợp để hỗ trợ sức khỏe tinh thần của họ.<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=300, use_cache=True)
        response = tokenizer.batch_decode(outputs)[0]
        
        # Extract assistant response (simple extraction)
        if "<|im_start|>assistant" in response:
            assistant_part = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in assistant_part:
                assistant_text = assistant_part.split("<|im_end|>")[0].strip()
            else:
                assistant_text = assistant_part.strip()
            print(f"\nAssistant: {assistant_text}")
        else:
            print(f"\nAssistant: {response}")
