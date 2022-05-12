import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import uuid
from flask import Flask, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename as sfn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from flask_ngrok import run_with_ngrok
'''
	Import Model & tf Here
'''
from model_new import *
import tensorflow as tf
from load_data import *


ALLOWED_EXTS = set(['.jpg', '.jpeg', '.png'])
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
PORT = 8080
INPUT_SHAPE = (331,331)

app = Flask(__name__)
run_with_ngrok(app)

'''
	Load the weghts & make graph global
'''
# md.load_weights("model.h5")
global graph
graph = tf.compat.v1.get_default_graph()
model=load_model()
model.summary()
#model.load_weights('bestweights.hdf5')

# App configuration
def configure(app):
	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
	app.config['RESULT_FOLDER'] = RESULT_FOLDER
	app.config['TEMPLATES_AUTO_RELOAD'] = True
	app.config['ENV'] = 'development'
	app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        # Error!
        return 'file not sent'

    file = request.files['file']
    
    if not file.filename:
        # Error!
        return 'filename error'

    _, file_ext = os.path.splitext(file.filename)
    
    if file_ext not in ALLOWED_EXTS:
        # Error!
        return 'file ext not allowed'

    filename = uuid.uuid4().hex + file_ext

    file.save(os.path.join('static', UPLOAD_FOLDER, filename))

    return redirect(url_for('result', uploaded_img=filename))

@app.route('/result', methods=['GET'])
def result():
    uploaded_img = os.path.join('static', UPLOAD_FOLDER, request.args['uploaded_img'])
    preprocessed_path = os.path.join('static', RESULT_FOLDER, request.args['uploaded_img'])
    app.logger.debug(uploaded_img)
    
    img = cv2.imread(uploaded_img)
    #app.logger.debug(img.shape)
    
    preprocessed,prediction = VizGradCAM(model,img,preprocessed_path)

    
    fin_img = np.expand_dims(preprocessed, 0)    #Resize to (1, img_dim)
    
    return render_template('result.html', 
                            uploaded_img=uploaded_img, 
                            preprocessed_img=preprocessed_path,
                            result=prediction)
if __name__ == "__main__":
    configure(app)
    app.run()