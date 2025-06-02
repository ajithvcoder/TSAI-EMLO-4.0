import httpx
import json

async def test_connection():
    with open("input.json") as f:
        payload = json.load(f)
    headers = {"Host": "timm-model-2-predictor.default.emlo.tsai", "Content-Type": "application/json"}
    # url = "http://pedantic_bardeen:8080/infer_sports"
    # a9b064a181f464079a66b39f0f8dc30a-527434120.ap-south-1.elb.amazonaws.com
    url = "http://a9b064a181f464079a66b39f0f8dc30a-527434120.ap-south-1.elb.amazonaws.com/v1/models/cat-classifier:predict"
    # url = ""
    try:
        async with httpx.AsyncClient() as client:
            #headers = {"Host": "cat-classifier.default.emlo.tsai", "Content-Type": "application/json"}
            response = await client.post(url, headers=headers, json=payload)
            print("Response:", response.json())
    except httpx.ConnectError as e:
        print("Connection failed:", e)

# Run the test
import asyncio
asyncio.run(test_connection())