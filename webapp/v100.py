VERSION_STR = 'v0.0.1'


import cv2
import base64
import requests
import numpy as np
from error import Error
from flask import Blueprint, request, jsonify
from random import shuffle

import os

blueprint = Blueprint(VERSION_STR, __name__)


FACE_CASCADE = cv2.CascadeClassifier("resources/cascades/haarcascades/haarcascade_frontalface_alt.xml")
EYE_CASCADE = cv2.CascadeClassifier("resources/cascades/haarcascades/haarcascade_eye.xml")


def base64_encode_image(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ret, image_buf = cv2.imencode('.jpg', image_bgr, (cv2.IMWRITE_JPEG_QUALITY, 40))
    image_str = base64.b64encode(image_buf)
    return 'data:image/jpeg;base64,' + image_str

def file_paths(path):
	#Creates list of directories for chosen number of catigories.
    files_path = []
    directories = [path+'/'+f+'/' for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    try: directories.remove(path+'/'+'.DS_Store'+'/')
    except: pass
    for dir_ in directories:
        temp_paths = [dir_+f for f in os.listdir(dir_)]
    	try: temp_paths.remove(dir_+'.DS_Store')
    	except: pass
    	files_path.extend(temp_paths)
    return files_path



def make_thumbnails(n_thumbnails):
    image_paths = file_paths('../../K9-Identifier/data/full_data')
    shuffle(image_paths)
    return [base64_encode_image(cv2.imread(img_path)) for img_path in image_paths[:n_thumbnails]]


def predict_breed(ID, image_rgb, n_preds, n_train_images, n_epochs, augment, mixed, return_image):
    #Save Image by ID for database
    ID = '1234567890'
    imgurl = 'https://github.com/CharlesLynn/K9-Identifier'
    probs = [
    {'prob': 0.9, 'breed':'german_shepherd'},
    {'prob': 0.6, 'breed': 'husky'},
    {'prob': 0.2, 'breed': 'labrador'}
    ]



    return ID, imgurl, probs


def feedback(ID, thumbs_up):
    return None

def obtain_images(request):
    '''
    All three routes below pass the image in the same way as one another.
    This function attempts to obtain the image, or it throws an error
    if the image cannot be obtained.
    '''

    if 'image_url' in request.args:
        image_url = request.args['image_url']
        try:
            response = requests.get(image_url)
            encoded_image_str = response.content
        except:
            raise Error(2873, 'Invalid `image_url` parameter')

    elif 'image_buf' in request.files:
        image_buf = request.files['image_buf']  # <-- FileStorage object
        encoded_image_str = image_buf.read()

    else:
        raise Error(35842, 'You must supply either `image_url` or `image_buf`')

    if encoded_image_str == '':
        raise Error(5724, 'You must supply a non-empty input image')

    encoded_image_buf = np.fromstring(encoded_image_str, dtype=np.uint8)
    decoded_image_bgr = cv2.imdecode(encoded_image_buf, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(decoded_image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


@blueprint.route('/predict', methods=['POST'])
def predict():
    # ID, Image, n_preds, n_samples, n_epochs, agument, mixed=False, return_image=False
    '''
    Predict
    Process an image of a dog and return the probabilities of breeds.
    Note: MOCK MODEL. Images do not need to be resized, but the model will likely use 150px by 150px
    ---
    tags:
      - v0.0.1 (Mock)

    responses:
      200:
        description: id, imgurl, probabilities
        schema:
          $ref: '#/definitions/predict'
      default:
        description: Unexpected error
        schema:
          $ref: '#/definitions/Error'

    parameters:

      - name: ID
        in: query
        description: ID associated with the predict and image. If none is provided one will be assigned.
        required: false
        type: string
      - name: image_url
        in: query
        description: The URL of an image that should be processed. If this field is not specified, you must pass an image via the `image_buf` form parameter.
        required: false
        type: string
      - name: image_buf
        in: formData
        description: An image that should be processed. This is used when you need to upload an image for processing rather than specifying the URL of an existing image. If this field is not specified, you must pass an image URL via the `image_url` parameter.
        required: false
        type: file
      - name: n_preds
        in: query
        description: Number of predictions to return. (default n_preds = 3)
        type: number
      - name: n_train_images
        in: query
        description: Number of images used to train the model. (default n_preds = MAX)
        type: integer
      - name: n_epochs
        in: query
        description: Number of epochs used to train the model. (default n_preds = OPTIMAL)
        type: integer
      - name: augment
        in: query
        description: Number of images used to train the model. (default augment = True)
        type: boolean
      - name: mixed
        in: query
        type: boolean
      - name: return_image
        in: query
        description: A boolean input flag (default=false) indicating whether or not to return the image.
        required: false
        type: boolean

    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
          id: predict
          type: object
          required:
            - probs
          properties:
            probs:
              description: Dictionary of predictions.
              schema:
                type: array
                items:
                  $ref: '#/definitions/probs'
            ID:
              type: string
              description: ID string optional, if not supplied one will be generated.
            return_image:
              type: string
              format: byte
              description: base64 encoded given image.
            imgurl:
              type: string
              description: url string.
      - schema:
          id: probs
          type: object
          required:
            - prob
            - breed
            - imgurl

          properties:
            prob:
              type: number
              description: the probability of this breed (e.g. 0.6)
            breed:
              type: string
              description: the name of the breed (e.g. 'german_shepherd')

    '''
    ID = request.args.get('ID', None)
    image_url = request.args.get('image_url', '')
    n_preds = int(request.args.get('n_preds', 3))
    n_train_images = int(request.args.get('n_train_images', None))
    n_epochs = int(request.args.get('n_epochs', None))
    augment = request.args.get('augment', True)
    return_image = request.args.get('return_image', False)
    mixed = request.args.get('mixed', False)
    image_rgb = obtain_images(request)

    predictinfo = predict_breed(ID, image_rgb, n_preds, n_train_images, n_epochs, augment, mixed, return_image)
    return jsonify(predictinfo)


@blueprint.route('/get_thumbnails', methods=['POST'])
def get_thumbnails():
    '''
    Returns images for website thumbnails.
    Returns images for website thumbnails. A list of n thumbnails, 150px x 150px. (Dims adjustable?)
    ---
    tags:
      - v0.0.1 (Mock)

    responses:
      200:
        description: Request for n images.
        schema:
          $ref: '#/definitions/get_thumbnails'
      default:
        description: Unexpected error
        schema:
          $ref: '#/definitions/Error'

    parameters:
      - name: n_images
        in: query
        description: Number of images to return.
        required: true
        type: int

    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
          id: get_thumbnails
          type: object
          required:
            - Images

          properties:
            Images:
              type: list
              description: list of base64 encoded given image strings.
    '''
    n_images = int(request.args.get('n_images', 3))
    return jsonify(make_thumbnails(n_images))


@blueprint.route('/feedback', methods=['POST'])
def feedback():
    '''
    Databasedes good/bad bool with Image, by ID.
    Databasedes good/bad bool with Image, by ID.
    ---
    tags:
      - v0.0.1 (Mock)



    parameters:
      - name: ID
        in: query
        description: ID string
        required: true
        type: string
      - name: thumbs_up
        in: query
        description: A boolean for thumbs-up/thumbs-down.
        required: true
        type: boolean

    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
          id: PhotoInfo
          type: object
          required:
            - faces
          properties:
            faces:
              description: an array of FaceInfo objects found in this image
              schema:
                type: array
                items:
                  $ref: '#/definitions/FaceInfo'
            annotated_image:
              type: string
              format: byte
              description: base64 encoded annotated image
    '''
    return jsonify({})


from app import app
app.register_blueprint(blueprint, url_prefix='/'+VERSION_STR)
