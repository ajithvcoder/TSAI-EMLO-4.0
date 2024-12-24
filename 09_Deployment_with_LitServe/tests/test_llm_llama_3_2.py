# client.py
from openai import OpenAI
import json
from typing import List, Dict

def run_benchmark(
    prompt: str,
    model_name: str,
    num_runs: int = 1,
    max_tokens: int = 250
) -> List[Dict]:
    """
    Run multiple benchmark tests and collect results
    """
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy-key"
    )
    
    results = {}
    
    
    for run_no in range(num_runs):
        output = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens= max_tokens,
            stream=False,
        )
        output_loaded = json.loads(output.choices[0].message.content)
        print(model_name, output_loaded["model_throughput"])
        results[run_no] = [output_loaded["model_throughput"], output_loaded["theoretical_max"]]
        
    return results

if __name__ == "__main__":
    test_prompt = "Explain about theory of relativity ?"
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_tokens = 250
    results = run_benchmark(test_prompt, model_name, num_runs=5, max_tokens=max_tokens)
    print(f"Benchmakring for {model_name} with max_tokens {max_tokens}")
    for key, value in results.items():
        print(f"Run no {key} - model_throughput(tokens/sec) - {value[0]} | theoretical_max - {value[1]} ")
    
    max_tokens = 500
    results = run_benchmark(test_prompt, model_name, num_runs=5, max_tokens=500)
    print(f"Benchmakring for {model_name} with max_tokens {max_tokens}")
    for key, value in results.items():
        print(f"Run no {key} - model_throughput(tokens/sec) - {value[0]} | theoretical_max - {value[1]} ")
