from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the style transfer model
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    img = img.convert('RGB')
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
    img = np.array(img)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.0
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    content_file = request.files.get('content_image')
    style_file = request.files.get('style_image')

    if not content_file or not style_file:
        return redirect(url_for('index'))

    # Save the uploaded files
    content_filename = secure_filename(content_file.filename)
    style_filename = secure_filename(style_file.filename)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
    content_file.save(content_path)
    style_file.save(style_path)

    
    content_image = load_img(content_path)
    style_image = load_img(style_path)

   
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

   
    result_image = tensor_to_image(stylized_image)
    result_filename = f"stylized_{content_filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    result_image.save(result_path)

    return redirect(url_for('show_result', filename=result_filename))

@app.route('/result/<filename>')
def show_result(filename):
    return render_template('result.html', filename=filename)

@app.route('/static/results/<filename>')
def display_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
