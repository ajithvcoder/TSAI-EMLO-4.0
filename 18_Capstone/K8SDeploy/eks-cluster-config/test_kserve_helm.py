import requests
import json
# cat-classifier
url = "http://a9b064a181f464079a66b39f0f8dc30a-527434120.ap-south-1.elb.amazonaws.com/v1/models/cat-classifier:predict"

with open("input.json") as f:
    payload = json.load(f)
headers = {"Host": "timm-model-2.default.emlo.tsai", "Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, json=payload)

print(response.headers)
print(response.status_code)
print(response.json())

# http://food-classifier.default.emlo.tsai