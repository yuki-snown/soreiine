import requests
import base64

url = 'http://localhost:5000/eval'

with open("api/test.jpg","rb") as f:
    binary = f.read()

img_base64 = base64.b64encode(binary)
img_str = img_base64.decode('utf-8')

response = requests.post(url, data={'img': img_str})
print(response.text)