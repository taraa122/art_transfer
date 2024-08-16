import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageEnhance


hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def load_img(path_to_img, max_dim=1024):
    """
    Load an image and resize it to have a maximum dimension of max_dim pixels.

    :param path_to_img: Path to the image file.
    :param max_dim: Maximum dimension (width or height) for the image.
    :return: Preprocessed image as a NumPy array.
    """
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = np.array(img)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.0
    return img

def tensor_to_image(tensor):
    """
    Convert a TensorFlow tensor to a PIL image.

    :param tensor: TensorFlow tensor.
    :return: PIL image.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def enhance_image(image):
    """
    Enhance the image by increasing contrast and sharpness.

    :param image: PIL image.
    :return: Enhanced PIL image.
    """
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2) 

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)  

    return image

def apply_style_transfer(content_path, style_path, output_dim=1024):
    """
    Apply style transfer to the given content and style images.

    :param content_path: Path to the content image.
    :param style_path: Path to the style image.
    :param output_dim: Maximum dimension for the output image.
    :return: Path to the stylized image.
    """
    
    content_image = load_img(content_path, max_dim=output_dim)
    style_image = load_img(style_path, max_dim=512)  # Smaller styles can still work well

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

   
    result_image = tensor_to_image(stylized_image)

    
    result_image = enhance_image(result_image)

  
    result_path = content_path.replace('uploads', 'results')
    result_image.save(result_path)

    return result_path
