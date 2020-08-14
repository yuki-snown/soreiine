import os, time
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from keras.layers import Input, Dense
from keras.models import Model
from efficientnet.keras import EfficientNetB4

app = Flask(__name__)

net = efficient_model()
net.load_weights("./b4-4.h5")

def efficient_model():
    input_layer = Input(shape=(300, 300, 3))
    efficient_net = EfficientNetB4(
        weights='noisy-student',
        include_top=False,
        input_tensor = input_layer,
        pooling='max')
    for layer in efficient_net.layers:
        layer.trainable = True
    x = efficient_net.output 
    x = Dense(units=128, activation='relu')(x)
    output = Dense(units=4, activation='softmax', name='class_output')(x)
    return Model(inputs=input_layer, outputs=output)

@app.route("/eval")
def eval():
    img = Image.open("test4.jpg")
    img = img.resize((300,300))
    img = np.array(img,dtype='float32') / 255.0
    img = np.expand_dims(img, 0)
    tmp = time.time()
    result = net.predict(img)
    tmp = time.time() - tmp
    data = {
        "result":result,
        "time":round(tmp,3)
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)