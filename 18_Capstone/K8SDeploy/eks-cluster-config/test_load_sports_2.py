import requests
import time
import concurrent.futures
import os
from statistics import mean
import argparse
from typing import List, Dict
import json
import matplotlib.pyplot as plt

def load_image() -> bytes:
    """Load the test image"""
    image_path = os.path.join(os.path.dirname(__file__), "dog.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        return image_file.read()

def make_request(url: str) -> Dict:
    """Make a single request to the API"""
    with open("input.json") as f:
        payload = json.load(f)
    headers_1 = {"Host": "timm-model-2.default.emlo.tsai", "Content-Type": "application/json"}
    
    # files = {"image": ("dog.jpg", image_data, "image/jpeg")}
    start_time = time.time()
    # response = requests.post(f"{url}/infer", files=files)
    response_1 = requests.request("POST", url, headers=headers_1, json=payload)
    end_time = time.time()
    # print(response_1)
    # print(response_1.json())

    return {
        "status_code": response_1.status_code,
        "response_time": end_time - start_time,
        "success": response_1.status_code == 200,
    }

def plot_results(results: List[Dict], output_file: str = "response_times_sports.png") -> None:
    """Plot and save response times from results"""
    response_times = [
        r["response_time"] for r in results if r["response_time"] is not None
    ]
    success_flags = [
        "Success" if r["success"] else "Failure" for r in results
    ]
    colors = ["green" if flag == "Success" else "red" for flag in success_flags]

    if not response_times:
        print("No response times to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(response_times)), response_times, c=colors, alpha=0.7)
    plt.title("API Response Time per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Response Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\nResponse time graph saved as '{output_file}'")

def run_load_test(
    url: str, num_requests: int, max_workers: int
) -> List[Dict]:
    """Run concurrent load test"""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(make_request, url) for _ in range(num_requests)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                
            except Exception as e:
                results.append(
                    {
                        "status_code": None,
                        "response_time": None,
                        "success": False,
                        "error": str(e),
                    }
                )
    # print(results)
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze test results"""
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    response_times = [r["response_time"] for r in successful_requests]

    if not response_times:
        return {
            "total_requests": len(results),
            "successful_requests": 0,
            "failed_requests": len(failed_requests),
            "avg_response_time": None,
            "requests_per_second": 0,
        }

    total_time = sum(response_times)
    return {
        "total_requests": len(results),
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "avg_response_time": mean(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "requests_per_second": len(successful_requests) / total_time,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Load test for image classification API"
    )
    parser.add_argument("--url", default="<http://localhost:8000>", help="API base URL")
    parser.add_argument(
        "--requests", type=int, default=100, help="Number of requests to make"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of concurrent workers"
    )
    args = parser.parse_args()

    print(
        f"\\nStarting load test with {args.requests} requests using {args.workers} workers"
    )
    print(f"Target URL: {args.url}\\n")

    try:
        # Load image once
        # image_data = load_image()

        # Run health check
        # health_response = requests.get(f"{args.url}/health")
        # print(f"Health check status: {health_response.status_code}")
        # print(f"Health check response: {health_response.json()}\\n")

        # Run load test
        start_time = time.time()
        results = run_load_test(args.url, args.requests, args.workers)
        total_time = time.time() - start_time

        # Analyze results
        analysis = analyze_results(results)
        plot_results(results)
        # Print results
        print("\\nTest Results:")
        print("-" * 50)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total requests: {analysis['total_requests']}")
        print(f"Successful requests: {analysis['successful_requests']}")
        print(f"Failed requests: {analysis['failed_requests']}")
        if analysis["avg_response_time"]:
            print(f"Average response time: {analysis['avg_response_time']*1000:.2f} ms")
            print(f"Min response time: {analysis['min_response_time']*1000:.2f} ms")
            print(f"Max response time: {analysis['max_response_time']*1000:.2f} ms")
        print(f"Requests per second: {analysis['requests_per_second']:.2f}")

    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main()

# python test_requests.py --requests 20 --workers 10
# k8s-default-classifi-18da2b317c-1355353865.ap-south-1.elb.amazonaws.com
# python test_requests.py --url https://k8s-default-classifi-18da2b317c-1355353865.ap-south-1.elb.amazonaws.com --requests 1 --workers 1
# python test_requests.py --url https:// k8s-prod-modelser-107450934f-1470860234.ap-south-1.elb.amazonaws.com --requests 1 --workers 1
# python3 test_load_sports.py --url "http://a250c292ee6114b8e9d6f23f6d8690bc-1957734331.ap-south-1.elb.amazonaws.com/v1/models/sports-classifier:predict" --requests 20 --workers 4