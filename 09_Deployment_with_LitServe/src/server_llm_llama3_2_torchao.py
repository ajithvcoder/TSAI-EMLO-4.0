# server.py
import torch
import litserve as ls
import re
import json
import time
from typing import List, Dict, Any, Generator
import numpy as np
import bitsandbytes as bnb
from transformers import (
    TorchAoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os

class LlamaBenchmark:
    def __init__(self, device):
        base_model = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if torch.cuda.get_device_capability()[0] >= 8:
            # !pip install -qqq flash-attn
            torch_dtype = torch.uint4
            attn_implementation = "flash_attention_2"
        else:
            torch_dtype = torch.uint4
            attn_implementation = "eager"

        
        quantization_config = TorchAoConfig("int4_weight_only", group_size=128,modules_to_not_convert=['quantized_model._orig_mod.lm_head','quantized_model._orig_mod.model.layers[10]']) 
        quantized_model = AutoModelForCausalLM.from_pretrained(base_model, 
                            device_map="auto", quantization_config=quantization_config,
                            attn_implementation=attn_implementation)
        quantized_model.generation_config.cache_implementation = "static"
        import torchao
        torchao.quantization.utils.recommended_inductor_config_setter()
        quantized_model = torch.compile(quantized_model, mode="max-autotune")
        model = quantized_model
        # self.modules = self.find_all_linear_names(model)
        # LoRA config
        # peft_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        #     target_modules=self.modules
        # )

        # model = get_peft_model(model, peft_config)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.uint4,
            device_map="auto",
        )

        # Store device info for benchmarking
        self.device = device
        self.theoretical_max = self._calculate_theoretical_max()
        self.last_benchmark = {}

    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16 bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def _calculate_theoretical_max(self) -> float:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            # https://www.baseten.co/blog/llm-transformer-inference-guide/
            """
            time/token = total number of bytes moved (the model weights) / accelerator memory bandwidth
            time/token = (2 * 1B) bytes / (300 GB/s) = 6.67 ms/token
            Tokens/Second = 150 tokens/second
            """
            time_per_token = (2*(10**9))/ (300* (10**9))
            gpu_specs = {
                "T4": int(round(1/time_per_token, 0))
            }
            for gpu, max_tokens in gpu_specs.items():
                if gpu in gpu_name:
                    return max_tokens
        return 0
        
    def apply_chat_template(self, messages: List[Any]) -> str:
        """Convert messages to model input format"""
        formatted_messages = []
        print("messages-", messages)
        messages = [
            {'role': 'system', 'content': 'You are a General knowledge assistant provide answers accordingly in few lines'},
            {"role": "user", "content": messages[0].content}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    
    def __call__(self, prompt: str, max_new_tokens = 256) -> tuple:
        """Run model inference with benchmarking"""
        start_time = time.perf_counter()
        outputs = self.pipe(prompt, max_new_tokens= max_new_tokens, do_sample=True)
        end_time = time.perf_counter()
        
        # Calculate metrics
        self.generation_time = end_time - start_time
        
        return prompt, outputs
    def decode_response(self, raw_response):
        cleaned_response = raw_response[1][0]["generated_text"].split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[1]
        return cleaned_response

    def decode_tokens(self, outputs: tuple) -> str:
        """Decode output tokens to text"""
        return outputs
    
    def get_benchmark_results(self, tokens_per_sec) -> Dict[str, float]:
        """Return the latest benchmark results"""
        self.efficiency = (tokens_per_sec / self.theoretical_max * 100) if self.theoretical_max > 0 else 0
        return self.theoretical_max, self.efficiency

class LlamaBenchmarkAPI(ls.LitAPI):
    def setup(self, device: str):
        device = "cuda"
        """Initialize the model"""
        self.model = LlamaBenchmark(device)

    def decode_request(self, request: Any) -> str:
        """Process the incoming request"""
        if not hasattr(request, 'messages'):
            raise ValueError("No messages provided")
        if request.model != "unsloth/Llama-3.2-1B-Instruct":
            raise Exception("Model not found")
        output = self.model.apply_chat_template(request.messages)
        self.max_tokens = request.max_tokens
        return output

    def predict(self, prompt: str, context: Any) -> Generator[tuple, None, None]:
        """Generate response"""
        yield self.model(prompt, self.max_tokens)

    def encode_response(self, outputs):
        """Format the response with benchmark results"""
        # print("outputs---")
        for output in outputs:
            generated_text = self.model.decode_response(output)
            tokens_generated = len(self.model.tokenizer.encode(generated_text, truncation=True))
            
            theoretical_max, efficiency = self.model.get_benchmark_results(float(tokens_generated/self.model.generation_time))
            # Count the tokens
            
            benchmark_results = {"gen_num_tokens": tokens_generated, "gen_time": self.model.generation_time,
                                 "model_throughput": float(tokens_generated/self.model.generation_time),
                                 "theoretical_max": theoretical_max , "efficiency": efficiency }
            response = {
                "role": "assistant",
                "content": json.dumps({"text": str(generated_text) , **benchmark_results })
            }
            # print("response-", response)
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