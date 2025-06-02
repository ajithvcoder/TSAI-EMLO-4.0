import requests
import json
# cat-classifier
url = "http://a775a10130ee74cd49b37e0b4b585f9e-1718843569.ap-south-1.elb.amazonaws.com/v1/models/dog-classifier:predict"

with open("input.json") as f:
    payload = json.load(f)
headers = {"Host": "food-classifier.default.emlo.tsai", "Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, json=payload)

print(response.headers)
print(response.status_code)
print(response.json())

# http://food-classifier.default.emlo.tsai