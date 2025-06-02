import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse

def send_single_request(payload, headers, url):
    """Send a single request and return the response metrics"""
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload)
        end_time = time.time()
        return {
            "status_code": response.status_code,
            "response_time": end_time - start_time,
            "success": response.status_code == 200,
        }
    except Exception as e:
        end_time = time.time()
        return {
            "status_code": 0,
            "response_time": end_time - start_time,
            "success": False,
            "error": str(e),
        }

def load_test(total_requests, concurrent_requests):
    """Execute load test with given parameters"""
    # Load configuration
    url = "http://ae394b43f53544d7baae6c66b87473ef-924312556.ap-south-1.elb.amazonaws.com/v1/models/imagenet-classifier:predict"
    headers = {
        "Host": "imagenet-classifier.default.emlo.tsai",
        "Content-Type": "application/json",
    }

    # Load payload
    with open("input.json") as f:
        payload = json.load(f)

    results = []
    start_time = time.time()

    print(f"\\nStarting load test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total requests: {total_requests}")
    print(f"Concurrent requests: {concurrent_requests}")
    print("-" * 50)

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [
            executor.submit(send_single_request, payload, headers, url)
            for _ in range(total_requests)
        ]

        # Collect results as they complete
        for future in futures:
            results.append(future.result())

    end_time = time.time()

    # Calculate statistics
    successful_requests = sum(1 for r in results if r["success"])
    failed_requests = total_requests - successful_requests
    total_time = end_time - start_time
    avg_response_time = sum(r["response_time"] for r in results) / len(results)

    # Print results
    print("\\nLoad Test Results:")
    print("-" * 50)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Average response time: {avg_response_time:.3f} seconds")
    print(f"Requests per second: {total_requests/total_time:.2f}")

def main():
    parser = argparse.ArgumentParser(
        description="Load Testing Tool for Image Classification API"
    )
    parser.add_argument(
        "-n",
        "--total-requests",
        type=int,
        default=100,
        help="Total number of requests to send (default: 100)",
    )
    parser.add_argument(
        "-c",
        "--concurrent-requests",
        type=int,
        default=30,
        help="Number of concurrent requests (default: 10)",
    )

    args = parser.parse_args()

    load_test(args.total_requests, args.concurrent_requests)

if __name__ == "__main__":
    main()
