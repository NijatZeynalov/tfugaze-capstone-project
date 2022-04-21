#!/usr/bin/env python

import os
from threading import Barrier
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from flask import jsonify

def prediction(predmodel):
    xtest = np.loadtxt('xtest.txt', dtype=int)
    ytest = np.loadtxt('ytest.txt', dtype=int)
    model = tf.keras.models.load_model(predmodel)
    y_pred = model.predict_classes(xtest)
    print(y_pred)
    acc = balanced_accuracy_score(ytest, y_pred)
    print(acc)
    return acc

# Initialize the Flask application

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['h5'])
app.config['MAX_CONTENT_LENGTH'] = 4 * 2048 * 2048  # 2MB

# If the file you are trying to upload is too big, you'll get this message
@app.errorhandler(413)
def request_entity_too_large(error):
    message = 'The file is too large.<br>'
    maxFileSizeKB = app.config['MAX_CONTENT_LENGTH']/(1024)
    message += "The biggest I can handle is " + str(maxFileSizeKB) + "KB"
    message += "<a href='" + url_for("index") + "'>Try again</a>"
    return message, 413

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# The root where we ask user to enter a file
@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the file upload
@app.route('/upload', methods=['POST'])

def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if not allowed_file(file.filename):
        message = "Sorry. Only files that end with one of these "
        message += "extensions is permitted: " 
        message += str(app.config['ALLOWED_EXTENSIONS'])
        message += "<a href='" + url_for("index") + "'>Try again</a>"
        return message
    elif not file:
        message = "Sorry. There was an error with that file.<br>"
        message += "<a href='" + url_for("index") + "'>Try again</a>"
        return message        
    else:
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        mabser = prediction(f'uploads/{filename}')

        if mabser<0.85:
            message = f"Model accuracy less than expected. Your score in out-of-bag dataset: {str(mabser)}"
        else:
            message = f"Congratulations. Your model score in out-of-bag dataset is: {str(mabser)}"
        return render_template('predict.html', result=message, )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__=='__main__':
    app.debug = True
    app.run()
