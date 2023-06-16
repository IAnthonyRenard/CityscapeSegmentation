#app.py
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, make_response, send_from_directory
#import urllib.request
import os
from werkzeug.utils import secure_filename
import pipeline
from PIL import Image
import numpy as np
import cv2




app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png'])#, 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Image non choisie')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('nom image: ' + filename)
        
        

        

        
        #########################PREDICTION DU MASQUE################################################
        #image ="C:/Users/Utilisateur/PROJET8/input/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/val/munster/munster_000034_000019_leftImg8bit.png"
        #image='C:/Users/Utilisateur/PROJET8/static/uploads/' + filename
        image=UPLOAD_FOLDER + filename
        
        #print(image)
        mask_t=pipeline.affichage_model_result(image)
        init_img = Image.fromarray((mask_t * 255).astype(np.uint8))
        init_img.save('static/uploads/mask_image.png')
        mask_filename = 'mask_image.png'
        print('nom mask: ' + mask_filename)

        
        
        flash(filename)
        print(filename)
        #print(mask_filename)
        return render_template('index.html',  filename=filename,mask_filename=mask_filename)
        #return send_file(mask_filename, mimetype='image/png')
        
        #########################FINPREDICTION DU MASQUE################################################
    
    else:
        flash('Le type autorisé est png')
        return redirect(request.url)
    
    

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/display/<mask_filename>')
def display_mask(mask_filename):
    return redirect(url_for('static', filename='uploads/' + mask_filename), code=301) #filename='uploads/' + mask_filename


@app.route('/', methods=['POST'])
def index():
    response = send_from_directory(directory='static/uploads', filename='mask_image.png')
    response.headers['my-custom-header'] = 'my-custom-status-0'
    print("MASQUE ENVOYEE")
    return response


if __name__ == "__main__":
    app.run()