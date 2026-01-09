# shared_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import threading
from accelerate import Accelerator
class SharedLlamaModel:
    
    def __init__(self):
        if True:
            self.model_name = 'Qwen/Qwen2.5-72B-Instruct'  # or "meta-llama/Llama-3.3-70B-Instruct"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            cache_dir = ""

            print(f"Loading model {self.model_name} on {self.device}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                token=""
            )
            
            # Add pad token if it doesn't exist
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            accelerator = Accelerator()
            self.device = accelerator.device
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir,
                token=""
            )
            
            self.model.gradient_checkpointing_enable()
            print("Model loaded successfully!")
    
    def format_chat_template(self, messages: List[Dict[str, str]]) -> str:
       
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.01, max_tokens: int = 800) -> str:
        
        formatted_prompt = self.format_chat_template(messages)
        
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            ).to(self.device)
        
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=3,#10,
                top_p=0.1,  
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        
        return response


global_model = SharedLlamaModel()