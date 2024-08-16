import functools
import os
import tensorflow as tf
import matplotlib.pylab as plt
from matplotlib import gridspec

def crop_center(image):
    ...

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    ...

def show_n(images, titles=('',)):
    ...
