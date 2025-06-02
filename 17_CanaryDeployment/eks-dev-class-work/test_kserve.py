import requests
import json
# cat-classifier
url = "http://acb6e8480c3e843b39e6b911100341bb-1050303067.ap-south-1.elb.amazonaws.com/v1/models/imagenet-vit:predict"

with open("input.json") as f:
    payload = json.load(f)
headers = {"Host": "imagenet-vit.default.emlo.tsai", "Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, json=payload)

print(response.headers)
print(response.status_code)
print(response.json())