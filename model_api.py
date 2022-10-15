"""http://web.univ-ubs.fr/lmba/lardjane/python/c4.pdf -> page 260"""

from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import segmentation_models as sm


app = Flask(__name__)


# Chargement du modèle
model = sm.Unet('vgg16', classes=8)
model.load_weights('unet_vgg16_aug_weights.h5')


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
    from waitress import serve
    serve(app)
