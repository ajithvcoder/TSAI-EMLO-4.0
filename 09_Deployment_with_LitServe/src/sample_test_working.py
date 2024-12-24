from transformers import (
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

from trl import  setup_chat_format

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# checkpoint = "meta-llama/Llama-3.2-1B"
# checkpoint = "meta-llama/Llama-2-7b-hf"
checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = "<ADD API KEY>"
# Initialize tokenizer and model
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,  token=HF_TOKEN, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     checkpoint,
#     torch_dtype=torch.bfloat16,
#     device_map=device,
#     token=HF_TOKEN
# )
print(torch.cuda.get_device_capability())
# exit()
if torch.cuda.get_device_capability()[0] >= 8:
    # !pip install -qqq flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
# Load model
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)

model = get_peft_model(model, peft_config)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

# toker = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
messages = [
    {'role': 'system', 'content': 'You are a General knowledge assistant provide answers accordingly in few lines'},
    {"role": "user", "content": "Who is Vincent van Gogh?"}
]
print('###### Default (yet Correct) Chat Template ######')
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print('###### Corrected Chat Template ######')

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True)

print(outputs[0]["generated_text"])

# outputs = model.generate(
#                 inputs,
#                 max_new_tokens=512,
#                 temperature=0.2,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id,
#                 # attention_mask=attention_mask
#             )

# chat_template = open('./chat_templates/llama-3-instruct.jinja').read()
# chat_template = chat_template.replace('    ', '').replace('\n', '')
# toker.chat_template = chat_template
# print(toker.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print(outputs)

