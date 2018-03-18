import requests

KERAS_REST_API_URL = 'http://localhost:5000/predict'
IMAGE_PATH = 'dog.jpg'

image = open(IMAGE_PATH, 'rb').read()
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()

if r['success']:
    for (i, result) in enumerate(r['predictions']):
        print('{}. {}'.format(i + 1, result['label']))
else:
    print('Request failed')