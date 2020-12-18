import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image as Image
from tensorflow.python.framework.errors_impl import UnknownError


def generate_image_embeddings(model_link, tf_image_url_lookup, image_shape, unique_item_ids):
    model = tf.keras.Sequential([
    hub.KerasLayer(model_link,
                   trainable=False)])
    model.build([None, image_shape[0], image_shape[1], 3])
    im_embeddings = {}
    for i, uri in enumerate(unique_item_ids):
        if i % 10 == 0:
            print(f"Downloading image {i}")
        url = tf_image_url_lookup.lookup(tf.constant(str.encode(uri)))
        image = read_image(url, image_shape)
        embedding = model(image)
        if isinstance(embedding.numpy()[0][0], np.float32):
            im_embeddings[uri] = embedding.numpy()[0]

    return im_embeddings


def read_image(file_url, image_shape):
    file_url_string = file_url.numpy().decode()

    if file_url_string == "nill":
        return np.zeros(shape=(1, image_shape[0], image_shape[1],3))+0.5

    if file_url_string.startswith('http') is False:
        file_url_string = 'https://' + file_url_string

    try:
        im = tf.keras.utils.get_file(file_url_string.replace('/','_'), file_url_string)
        im = Image.open(im).resize(image_shape)
        im = np.array(im)/255.0
        try:
            im = im[:,:,:3]
        except (IndexError, UnknownError):
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        return im[np.newaxis, ...]
    except:
        return np.zeros(shape=(1, image_shape[0], image_shape[1],3))+0.5