import requests

url = "http://localhost:5003/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}

response = requests.post(url, json=data)
print("Response:", response.json())
