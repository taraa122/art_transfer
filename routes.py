from flask import render_template, request, jsonify
from app import app
from app.style_transfer import process_images
import os

UPLOAD_FOLDER = 'app/static/uploads'
RESULT_FOLDER = 'app/static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    content_file = request.files['content']
    style_file = request.files['style']

    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
    content_file.save(content_path)
    style_file.save(style_path)

    result_image = process_images(content_path, style_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
    result_image.save(result_path)

    return jsonify({'result_image': result_path}), 200
