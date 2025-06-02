import requests
import json
# cat-classifier
url = "http://ae394b43f53544d7baae6c66b87473ef-924312556.ap-south-1.elb.amazonaws.com/v1/models/imagenet-classifier:predict"

with open("input.json") as f:
    payload = json.load(f)
headers = {"Host": "imagenet-classifier.default.emlo.tsai", "Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, json=payload)

print(response.headers)
print(response.status_code)
print(response.json())

# http://food-classifier.default.emlo.tsai