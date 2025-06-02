import httpx
import requests
import logging

# print("Loading ImageNet categories")
# url = "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
# categories = requests.get(url).text.strip().split("\n")
# print(categories)
# print(f"Loaded {len(categories)} categories")

files = {'image': open('dog.jpg', 'rb')}
response = httpx.post("http://localhost:8080/infer", files=files)

print(response)