import requests

url = 'http://127.0.0.1:5000/predict'
files = {'file': open('data/original/jordan4-retro-griss.png', 'rb')}

response = requests.post(url, files=files)

print(response.text)