"""
Conversation management and turn handling.
"""

import re
import threading
import time

from config import (
    USER, ASSISTANT, END_SESSION_TOKEN, 
    GEMINI_CLIENT_TIMEOUT, NUMBER_OF_TURNS_PER_CONVERSATION, ATTITUDES
)
from prompts import CLIENT_PROMPT_TEMPLATE
from evaluator import evaluate_conversation_with_timeout, display_scores


def build_conversation_turns(user_turns, bot_turns):
    """Build a formatted conversation string for the model using Vietnamese role labels."""
    min_turns = min(len(user_turns), len(bot_turns))
    convo = ''
    
    for u, b in zip(user_turns[:min_turns], bot_turns[:min_turns]):
        convo += f'<|im_start|>{USER}\n{u}<|im_end|>\n<|im_start|>{ASSISTANT}\n{b}<|im_end|>\n'
    
    # Add the last user turn if it exists and hasn't been paired with a bot turn yet
    if len(user_turns) > len(bot_turns):
        convo += f'<|im_start|>{USER}\n{user_turns[len(bot_turns)]}<|im_end|>\n<|im_start|>{ASSISTANT}\n'
    
    return convo


def gemini_generate_user_reply_with_timeout(conversation_history_string, gemini_model, 
                                            system_prompt, timeout=GEMINI_CLIENT_TIMEOUT):
    """Generate a user reply using Gemini with timeout protection."""
    user_reply = [None]
    error_message = [None]
    completed = [False]
    
    def get_user_reply():
        try:
            prompt = f"""{system_prompt}

Lịch sử hội thoại:
{conversation_history_string}

Dựa trên lịch sử hội thoại trên, hãy đóng vai trò là "{USER}" và đưa ra phản hồi tiếp theo của bạn.
Chỉ trả lời dưới vai trò "{USER}" và chỉ cung cấp nội dung của phản hồi đó.
{USER}:"""
            
            response = gemini_model.generate_content(prompt)
            user_reply[0] = response
        except Exception as e:
            error_message[0] = str(e)
        finally:
            completed[0] = True
    
    # Start thread for API call
    api_thread = threading.Thread(target=get_user_reply)
    api_thread.daemon = True
    api_thread.start()
    
    # Wait with timeout
    start_time = time.time()
    while not completed[0] and time.time() - start_time < timeout:
        time.sleep(0.5)
    
    if not completed[0]:
        print(f"⚠️ Warning: Gemini API call for client response timed out after {timeout} seconds")
        return "Xin lỗi, tôi cần thêm thời gian để suy nghĩ về vấn đề của mình."
    elif error_message[0]:
        print(f"❌ Error from Gemini API during client response generation: {error_message[0]}")
        return "Tôi vẫn cảm thấy lo lắng về vấn đề của mình."
    else:
        reply = user_reply[0]
        # Clean up role label if model added it
        reply = re.sub(rf"^{USER}:\s*", "", reply).strip()
        return reply


def format_conversation_for_evaluation(user_turns, bot_turns):
    """Format the conversation for evaluation."""
    full_conversation_text = ""
    min_len_conv = min(len(user_turns), len(bot_turns))
    
    for i in range(min_len_conv):
        full_conversation_text += f"Người dùng: {user_turns[i]}\n\n"
        full_conversation_text += f"Cố vấn: {bot_turns[i]}\n\n"
    
    # Add the last user turn if the conversation ended with the user speaking
    if len(user_turns) > len(bot_turns):
        full_conversation_text += f"Người dùng: {user_turns[len(bot_turns)]}\n\n"
    
    return full_conversation_text


def run_conversation_evaluation(initial_prompt, intake_form, attitude, 
                                counselor_model, gemini_model, 
                                num_turns=NUMBER_OF_TURNS_PER_CONVERSATION):
    """Run a complete conversation evaluation with the given initial prompt."""
    if counselor_model.model is None or gemini_model.model is None:
        print("Skipping conversation due to model loading errors.")
        return {
            "initial_prompt": initial_prompt,
            "attitude": attitude,
            "user_turns": [initial_prompt],
            "bot_turns": [],
            "evaluation_text": "Skipped due to model errors.",
            "scores": None
        }
    
    user_turns = [initial_prompt]
    bot_turns = []
    
    # Build client prompt
    client_prompt = CLIENT_PROMPT_TEMPLATE.format(intake_form, attitude, ATTITUDES[attitude])
    
    print(f"\n=== Bắt đầu hội thoại mới (Initial Prompt: '{initial_prompt[:50]}...'; Attitude: {attitude}) ===")
    
    # Run the conversation
    for turn in range(num_turns):
        print(f'\n--- Cố vấn (Turn {turn+1}) ---')
        
        # Build conversation history for counselor
        conversation_history_for_counselor = build_conversation_turns(user_turns, bot_turns)
        
        # Generate counselor response
        bot_reply = counselor_model.generate_response(user_turns[-1], conversation_history_for_counselor)
        bot_turns.append(bot_reply)
        print(bot_reply)
        
        # Get client response from Gemini, but only if not the last turn
        if turn < num_turns - 1:
            print(f'\n--- Người dùng (Turn {turn+2}) ---')
            
            # Build conversation history for Gemini client
            conversation_history_for_gemini = build_conversation_turns(user_turns, bot_turns)
            
            user_reply = gemini_generate_user_reply_with_timeout(
                conversation_history_for_gemini,
                gemini_model,
                system_prompt=client_prompt
            )
            user_turns.append(user_reply)
            print(user_reply)
            
            # Check for end session token
            if END_SESSION_TOKEN in user_reply:
                print(f"\n--- Kết thúc phiên (Signal Detected Turn {turn+2}) ---")
                # Clean the token from the final user message
                user_turns[-1] = user_reply.replace(END_SESSION_TOKEN, "").strip()
                break
        
        time.sleep(0.2)
    
    # Evaluate conversation
    full_conversation_text = format_conversation_for_evaluation(user_turns, bot_turns)
    
    print("\n=== Đánh giá toàn bộ hội thoại ===")
    
    gemini_eval_text, conversation_scores = evaluate_conversation_with_timeout(
        gemini_model,
        intake_form,
        attitude,
        ATTITUDES[attitude],
        full_conversation_text
    )
    
    print("\n--- Gemini Evaluation Text ---")
    print(gemini_eval_text)
    
    # Display scores
    if conversation_scores:
        display_scores(conversation_scores, f"Đánh giá hội thoại: '{initial_prompt[:50]}...'")
    
    return {
        "initial_prompt": initial_prompt,
        "attitude": attitude,
        "num_turns": len(bot_turns),
        "user_turns": user_turns,
        "bot_turns": bot_turns,
        "evaluation_text": gemini_eval_text,
        "scores": conversation_scores
    }
