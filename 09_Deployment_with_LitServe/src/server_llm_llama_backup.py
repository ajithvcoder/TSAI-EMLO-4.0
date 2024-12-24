# server.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import litserve as ls
import time
from typing import List, Dict, Any, Generator
import numpy as np

class LlamaBenchmark:
    def __init__(self, device):
        # checkpoint = "NousResearch/Llama-3.2-1B"
        checkpoint = "meta-llama/Llama-2-7b-hf"
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Set chat template
        self.chat_template = """<s>[INST] {} [/INST] """
        
        self.model = torch.compile(self.model)
        self.model.eval()
        
        # Store device info for benchmarking
        self.device = device
        self.theoretical_max = self._calculate_theoretical_max()
        self.last_benchmark = {}
        
    def _calculate_theoretical_max(self) -> float:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_specs = {
                "A100": 275000,
                "V100": 140000,
                "T4": 45000,
            }
            for gpu, max_tokens in gpu_specs.items():
                if gpu in gpu_name:
                    return max_tokens
        return 0
        
    def format_chat_message(self, message: Any) -> str:
        """Format a single chat message"""
        try:
            # Handle both dict-like and object-like message formats
            content = message.content if hasattr(message, 'content') else message['content']
            role = message.role if hasattr(message, 'role') else message['role']
            
            if role == "system":
                return f"System: {content}\n"
            elif role == "user":
                return f"User: {content}\n"
            elif role == "assistant":
                return f"Assistant: {content}\n"
            return content
        except (AttributeError, KeyError):
            return str(message)

    def apply_chat_template(self, messages: List[Any]) -> str:
        """Convert messages to model input format"""
        formatted_messages = []
        
        # Format each message
        for message in messages:
            formatted_messages.append(self.format_chat_message(message))
        
        # Combine messages and apply template
        combined_message = "".join(formatted_messages)
        return self.chat_template.format(combined_message)
    
    def __call__(self, prompt: str) -> tuple:
        """Run model inference with benchmarking"""
        # Tokenize
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Benchmark generation
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.perf_counter()
        
        # Calculate metrics
        generation_time = end_time - start_time
        num_tokens = outputs.shape[1] - inputs.shape[1]
        tokens_per_sec = num_tokens / generation_time
        
        # Store benchmark results
        self.last_benchmark = {
            "tokens_generated": num_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_sec,
            "theoretical_max": self.theoretical_max,
            "efficiency": (tokens_per_sec / self.theoretical_max * 100) if self.theoretical_max > 0 else 0
        }
        
        return inputs, outputs
    
    def decode_tokens(self, outputs: tuple) -> str:
        """Decode output tokens to text"""
        inputs, generate_ids = outputs
        new_tokens = generate_ids[:, inputs.shape[1]:]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    
    def get_benchmark_results(self) -> Dict[str, float]:
        """Return the latest benchmark results"""
        return self.last_benchmark

class LlamaBenchmarkAPI(ls.LitAPI):
    def setup(self, device: str):
        """Initialize the model"""
        self.model = LlamaBenchmark(device)

    def decode_request(self, request: Any) -> str:
        """Process the incoming request"""
        if not hasattr(request, 'messages'):
            raise ValueError("No messages provided")
        print("request-", request)
        return self.model.apply_chat_template(request.messages)

    def predict(self, prompt: str, context: Any) -> Generator[tuple, None, None]:
        """Generate response"""
        yield self.model(prompt)

    def encode_response(self, outputs: Generator[tuple, None, None]) -> Generator[Dict[str, str], None, None]:
        """Format the response with benchmark results"""
        print(outputs)
        responses = []
        for output in outputs:
            response = {
                "role": "assistant",
                "content": self.model.decode_tokens(output),
                "benchmark": self.model.get_benchmark_results()
            }
            print("response-", response)
            yield response

if __name__ == "__main__":
    api = LlamaBenchmarkAPI()
    server = ls.LitServer(
        api,
        spec=ls.OpenAISpec(),
        accelerator="gpu",
        workers_per_device=1
    )
    server.run(port=8000)