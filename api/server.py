import os, time
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS
from model import efficient_model
import tensorflow as tf

app = Flask(__name__)
CORS(app)

global model, graph
model = efficient_model()
model.load_weights("./b4-4.h5")
graph = tf.get_default_graph()

@app.route("/eval")
def eval():
    img = Image.open("test4.jpg")
    img = img.resize((300,300))
    img = np.array(img,dtype='float32') / 255.0
    img = np.expand_dims(img, 0)
    tmp = time.time()
    with graph.as_default():
        result = model.predict(img)
    tmp = time.time() - tmp
    data = {
        "result":result.tolist()[0],
        "time":round(tmp,3)
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run()