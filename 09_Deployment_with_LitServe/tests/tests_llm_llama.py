# client.py
from openai import OpenAI
import json
from typing import List, Dict

def run_benchmark(
    prompt: str,
    num_runs: int = 1
) -> List[Dict]:
    """
    Run multiple benchmark tests and collect results
    """
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy-key"
    )
    
    results = []
    for i in range(num_runs):
        output = client.chat.completions.create(
            model="unsloth/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens= 250,
            stream=False,
        )
        benchmark_results = ""
        content = ""
        # print("stream")
        # print(stream)
        print(output.choices[0].message.content)
        output_loaded = json.loads(output.choices[0].message.content)
        # for chunk in stream:
        #     print(chunk)
        #     if chunk.choices[0].delta.content is not None:
        #         # print(chunk.choices[0].delta.content, end="")
        #         content = json.loads(chunk.choices[0].delta.content)
        #     if hasattr(chunk.choices[0].delta, "benchmark"):
        #         benchmark_results = chunk.choices[0].delta.benchmark
        #         # Process benchmark
        #     else:
        #         print("Benchmark attribute is not present.")
            # if "benchmark" in chunk.choices[0].delta and chunk.choices[0].delta.benchmark is not None:
            #     benchmark_results += chunk.choices[0].delta.benchmark
                # print(chunk.choices[0].delta.content, end="")
        print("########################################")
        print("only content")
        # print("content-", content)
        # print("benchmark_results-", benchmark_results)
        print(output_loaded["text"])
        print(output_loaded["gen_num_tokens"])
        print(output_loaded["gen_time"])
        # response_text = ""
        # for chunk in stream:
        #     if chunk.choices[0].delta.content is not None:
        #         response_text += chunk.choices[0].delta.content
        # print("choices-", chunk.choices)
        # # Extract benchmark results from the response
        # benchmark_data = json.loads(chunk.choices[0].delta.benchmark)
        # results.append(benchmark_data)
        
        # print(f"\nRun {i+1} Results:")
        # print(f"Tokens/sec: {benchmark_data['tokens_per_second']:.2f}")
        # print(f"Theoretical max: {benchmark_data['theoretical_max']}")
        # print(f"Efficiency: {benchmark_data['efficiency']:.2f}%")
        
    return results

if __name__ == "__main__":
    test_prompt = "Explain about theory of relativity in 2 lines ?"
    results = run_benchmark(test_prompt)
    
    # Calculate average metrics
    # avg_tokens_per_sec = sum(r['tokens_per_second'] for r in results) / len(results)
    # avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
    
    # print("\nAverage Benchmark Results:")
    # print(f"Average tokens/sec: {avg_tokens_per_sec:.2f}")
    # print(f"Average efficiency: {avg_efficiency:.2f}%")