"""
Evaluation and scoring logic for CBT conversations.
"""

import re
import threading
import time
import numpy as np

from config import CRITERIA, GEMINI_EVAL_TIMEOUT
from prompts import EVALUATION_PROMPT_TEMPLATE, CBT_EVALUATION_CRITERIA


def extract_scores_from_evaluation(evaluation_text):
    """Extract numerical scores (0-6) from the evaluation text for each criterion."""
    scores = {key: None for key in CRITERIA.keys()}
    
    for criterion_key, display_name in CRITERIA.items():
        patterns = [
            rf"{re.escape(display_name)}\s*[:\s]*([0-6])(?:/6|\s|$|\.)",
            rf"{re.escape(display_name)}\s*[:\s]*([0-6])",
            rf"\*\*{re.escape(display_name)}\*\*\s*[:\s]*([0-6])(?:/6|\s|$|\.)",
            rf"[0-6]\.[\s]*\*\*{re.escape(display_name)}\*\*\s*[:\s]*([0-6])(?:/6|\s|$|\.)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                try:
                    extracted_score = int(match.group(1))
                    scores[criterion_key] = extracted_score
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse valid integer score for {display_name} from match: '{match.group(0)}'.")
                break
    
    return scores


def evaluate_conversation_with_timeout(gemini_model, intake_form, attitude, attitude_desc, 
                                       conversation_text, timeout=GEMINI_EVAL_TIMEOUT):
    """Evaluate conversation using Gemini with timeout protection."""
    if gemini_model is None:
        print("Error: Gemini model not configured. Skipping evaluation.")
        return "Evaluation skipped: Gemini not configured.", None
    
    if not conversation_text:
        print("Error: No conversation provided for evaluation.")
        return "Evaluation skipped: No conversation provided.", None
    
    # Build evaluation prompt
    eval_prompt_text = EVALUATION_PROMPT_TEMPLATE.format(
        intake_form, 
        attitude, 
        attitude_desc,
        conversation_text, 
        CBT_EVALUATION_CRITERIA
    )
    
    evaluation_result = [None, None]  # [text, scores]
    error_message = [None]
    completed = [False]
    
    def get_evaluation():
        try:
            eval_text = gemini_model.generate_content(eval_prompt_text)
            scores = extract_scores_from_evaluation(eval_text)
            evaluation_result[0] = eval_text
            evaluation_result[1] = scores
        except Exception as e:
            error_message[0] = str(e)
        finally:
            completed[0] = True
    
    # Start thread for API call
    eval_thread = threading.Thread(target=get_evaluation)
    eval_thread.daemon = True
    eval_thread.start()
    
    # Wait with timeout
    start_time = time.time()
    while not completed[0] and time.time() - start_time < timeout:
        time.sleep(0.5)
    
    if not completed[0]:
        print(f"⚠️ Warning: Evaluation API call timed out after {timeout} seconds")
        return "Evaluation timed out. Continuing without evaluation.", None
    elif error_message[0]:
        print(f"❌ Error during evaluation: {error_message[0]}")
        return f"Evaluation failed due to error: {error_message[0]}", None
    else:
        return evaluation_result[0], evaluation_result[1]


def display_scores(scores, title="Evaluation Scores"):
    """Display scores (0-6) in a nice format without visualization."""
    if scores is None or all(v is None for v in scores.values()):
        print("No valid scores to display.")
        return
    
    print(f"\n--- {title} ---")
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    
    for criterion, score in valid_scores.items():
        print(f"{CRITERIA.get(criterion, criterion)}: {score}/6")
    
    if valid_scores:
        score_values = list(valid_scores.values())
        avg_score = np.mean(score_values)
        min_score = np.min(score_values)
        max_score = np.max(score_values)
        std_dev = np.std(score_values)
        
        print(f"Điểm trung bình: {avg_score:.2f}/6")
        print(f"Điểm thấp nhất: {min_score}/6")
        print(f"Điểm cao nhất: {max_score}/6")
        print(f"Độ lệch chuẩn: {std_dev:.2f}")


def aggregate_results(conversation_results):
    """Aggregate evaluation results across all conversations."""
    all_criteria_scores = {criterion: [] for criterion in CRITERIA.keys()}
    
    for result in conversation_results:
        if result.get('scores'):
            for criterion, score in result['scores'].items():
                if score is not None:
                    all_criteria_scores[criterion].append(score)
    
    # Calculate statistics
    criteria_stats = {}
    for criterion, scores in all_criteria_scores.items():
        if scores:
            criteria_stats[criterion] = {
                "mean": np.mean(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "std": np.std(scores)
            }
        else:
            criteria_stats[criterion] = {
                "mean": 0.0, 
                "min": None, 
                "max": None, 
                "std": 0.0
            }
    
    # Calculate overall average
    total_average = sum(stats["mean"] for stats in criteria_stats.values()) / len(criteria_stats) \
                    if criteria_stats and any(stats["mean"] is not None for stats in criteria_stats.values()) else 0.0
    
    return criteria_stats, total_average


def print_aggregate_results(criteria_stats, total_average, num_evaluated):
    """Print aggregated evaluation results."""
    print("\n=== Tổng hợp kết quả đánh giá các hội thoại ===")
    print(f"Số lượng hội thoại đã hoàn thành đánh giá: {num_evaluated}")
    
    for criterion, stats in criteria_stats.items():
        display_min = stats["min"] if stats["min"] is not None else 'N/A'
        display_max = stats["max"] if stats["max"] is not None else 'N/A'
        print(f"{CRITERIA.get(criterion, criterion)}: Trung bình: {stats['mean']:.2f}/6, "
              f"Thấp nhất: {display_min}/6, Cao nhất: {display_max}/6, "
              f"Độ lệch chuẩn: {stats['std']:.2f}")
    
    print(f"Điểm trung bình tổng thể: {total_average:.2f}/6")
