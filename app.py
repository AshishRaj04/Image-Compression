import os
from flask import Flask, request, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
import io
from kmeans_compression import startCompressing

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compress', methods=['POST'])
def compress_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']

    image_path = "temp.png"
    file.save(image_path)

    original_image = plt.imread(image_path)

    os.remove(image_path)

    print("The shape of the image is: ", original_image.shape)

    X_img = np.reshape(original_image, (-1, original_image.shape[2]))

    compressed_image_array = startCompressing(X_img)

    X_recovered = np.reshape(compressed_image_array, original_image.shape)
    compressed_image = X_recovered.astype('uint8')

    img_io = io.BytesIO()
    plt.imsave(img_io, compressed_image, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
