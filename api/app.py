import os, time, io, base64
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
from model import efficient_model
import tensorflow as tf

app = Flask(__name__)
CORS(app)

global model, graph
model = efficient_model()
model.load_weights("api/b4-4.h5")
graph = tf.get_default_graph()

@app.route("/eval", methods=["POST"])
def eval():
    img = base64.b64decode(request.form['img'])
    img_binarystream = io.BytesIO(img)
    img = Image.open(img_binarystream)
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