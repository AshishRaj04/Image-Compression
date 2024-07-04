from flask import Flask, request, render_template, send_file
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from kmeans_compression import compress_image_kmeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    image = Image.open(file.stream)
    image_array = np.array(image)

    k = 16  # Number of colors for K-means compression
    compressed_image_array = compress_image_kmeans(image_array, k)
    
    compressed_image = Image.fromarray(compressed_image_array)
    img_io = io.BytesIO()
    compressed_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
