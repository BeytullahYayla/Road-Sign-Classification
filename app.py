from json import load
from keras.models import load_model
import requests 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


url = 'https://www.arthipo.com/image/cache/catalog/poster/car/pcars240-engebeli-yol-isareti-bumpy-road-sign-600x315w.png'
model=load_model('model.h5')
model.summary()

def cvrt_grayscale(img):
  img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  return img

def equalize(img):
  img=cv2.equalizeHist(img)
  return img

def preprocessing(img):
  grayscale_image=cvrt_grayscale(img)
  equalized_image=equalize(grayscale_image)
  normalized_image=equalized_image/255
  return normalized_image

r=requests.get(url,stream=True)
img=Image.open(r.raw)

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
img = img.reshape(1, 32, 32, 1)

predicted_image=model.predict(img)
predicted_class=np.argmax(predicted_image,axis=1)
print("predicted-sign"+str(predicted_class))