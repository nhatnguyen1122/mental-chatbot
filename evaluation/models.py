"""
Model loading and response generation functions.
"""

import os
import re
import torch
import google.generativeai as genai
from unsloth import FastLanguageModel

from config import (
    FINETUNED_MODEL_PATH, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT,
    GEMINI_MODEL_NAME, USER, ASSISTANT, MAX_NEW_TOKENS,
    TEMPERATURE, TOP_P, MIN_P
)
from prompts import COUNSELOR_PROMPT_TEMPLATE


class CounselorModel:
    """Wrapper for the fine-tuned counselor model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load the fine-tuned model."""
        print("Loading fine-tuned counselor model...")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=FINETUNED_MODEL_PATH,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=DTYPE,
                load_in_4bit=LOAD_IN_4BIT,
            )
            FastLanguageModel.for_inference(self.model)
            print("Counselor model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading counselor model: {e}")
            return False
    
    def generate_response(self, current_user_input, conversation_history, max_new_tokens=MAX_NEW_TOKENS):
        """Generate a response from the counselor model."""
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded."
        
        # Build full prompt
        full_prompt = COUNSELOR_PROMPT_TEMPLATE.format(
            conversation_history + f'<|im_start|>{USER}\n{current_user_input}<|im_end|>\n<|im_start|>{ASSISTANT}\n<think>\n\n</think>\n\n'
        )
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        ).to(self.model.device)
        
        # Ensure attention mask
        if inputs.get("attention_mask") is None:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).to(self.model.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            min_p=MIN_P,
        )
        
        # Validate output
        if not isinstance(outputs, torch.Tensor) or outputs.shape[1] <= input_length:
            print(f"Warning: Model generated empty or invalid output.")
            return "Error: Model failed to generate a valid response (empty output)."
        
        # Decode
        decoded_output = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False)
        
        if not isinstance(decoded_output, str) or not decoded_output:
            print(f"Warning: Decoded output is invalid or empty.")
            return "Error: Model generated an invalid or empty response."
        
        # Extract response content
        response = self._extract_assistant_response(decoded_output)
        
        # Clean up tags
        response = re.sub(r'<\|im_end\|>$', '', response).strip()
        response = re.sub(r'^<\|im_start\|>.*?\n', '', response).strip()
        
        return response
    
    def _extract_assistant_response(self, decoded_output):
        """Extract the assistant's response from decoded output."""
        assistant_tag_start = decoded_output.rfind(f"<|im_start|>{ASSISTANT}\n")
        
        if assistant_tag_start != -1:
            content_start = assistant_tag_start + len(f"<|im_start|>{ASSISTANT}\n")
            end_tag_start = decoded_output.find("<|im_end|>", content_start)
            
            if end_tag_start != -1:
                return decoded_output[content_start:end_tag_start].strip()
            else:
                return decoded_output[content_start:].strip()
        else:
            return decoded_output.strip()


class GeminiModel:
    """Wrapper for Gemini API (client simulator and evaluator)."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = None
    
    def load(self):
        """Configure Gemini API."""
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not set. Gemini features will be unavailable.")
            return False
        
        print("Configuring Gemini API...")
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print(f"Gemini model ({GEMINI_MODEL_NAME}) configured successfully.")
            return True
        except Exception as e:
            print(f"Error configuring Gemini model: {e}")
            return False
    
    def generate_content(self, prompt):
        """Generate content using Gemini."""
        if self.model is None:
            raise RuntimeError("Gemini model not loaded.")
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
