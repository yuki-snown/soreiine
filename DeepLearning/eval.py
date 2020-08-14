import os, time
import numpy as np
from PIL import Image
from model import efficient_model

def main():
    net = efficient_model()
    net.load_weights("./b4-4.h5") # weight path
    img = Image.open("test4.jpg") # picture path
    img = img.resize((300,300))
    img = np.array(img,dtype='float32') / 255.0
    img = np.expand_dims(img, 0)
    tmp = time.time()
    result = net.predict(img)
    tmp = time.time() - tmp
    print(result)
    print(tmp)

if __name__ == "__main__":
    main()