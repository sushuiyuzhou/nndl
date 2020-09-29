import flask
from flask import Flask, render_template, request, url_for

import base64
import numpy as np
import cv2
import network

# Initialize the useless part of the base64 encoded image.
init_Base64 = 21

NET = network.load("trained.out")

# Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')


# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

# First route : Render the initial drawing template
@app.route('/')
def home():
    return render_template('draw.html')


# Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    draw = request.form['url']

    # Removing the useless part of the url.
    draw = draw[init_Base64:]

    # Decoding
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # Resizing and reshaping to keep the ratio.
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    vect = np.asarray(resized, dtype="uint8")

    result = NET.evaluate(vect.reshape(784, 1).astype('float32'))
    return render_template('results.html', prediction=str(result))


if __name__ == '__main__':
    app.run(debug=True)
