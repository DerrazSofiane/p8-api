"""http://web.univ-ubs.fr/lmba/lardjane/python/c4.pdf -> page 260"""

from flask import Flask, request, jsonify, Response
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image



app = Flask(__name__)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


score_IoU = tf.keras.metrics.OneHotMeanIoU(
    num_classes=8,
    name='mean_IoU')

# Chargement du modèle
model = tf.keras.models.load_model(
    'unet_vgg16_aug.h5',
    custom_objects={'mean_IoU': score_IoU, 'dice_coeff': dice_coeff})


# defining a route for only post requests
@app.route('/predict', methods=['POST'])
def predict():
    response = {}
    try:
        r = request
        # convert string of image data to uint8
        nparr = np.frombuffer(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # build a response dict to send back to client
        response = {'message': 'image received. size={}x{}'.format(
            img.shape[1], img.shape[0])}
        
        img = img/255
        x = cv2.resize(img, (512, 256))
        pred = model.predict(np.expand_dims(x, axis=0))
        pred_mask = np.argmax(pred, axis=-1)
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_mask = np.squeeze(pred_mask)

        # creating a response object
        # storing the model's prediction in the object
        response['prediction'] =  pred_mask.tolist()
    except Exception as e:
        response['error'] = e

    # returning the response object as json
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
