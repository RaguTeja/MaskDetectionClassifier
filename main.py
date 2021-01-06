from flask import Flask,render_template,request           # To develop web application
from tensorflow.keras.models import load_model            # To load the model
import os                                                 # path handling
from werkzeug.utils import secure_filename
from predict import Predict                               # prediction module


app=Flask(__name__)                                       # Initializing the flask application
model=load_model('model_vgg16.h5')                        # loaded the model


@app.route('/')                                           # Home page
def home():
    return render_template('index.html')                  # Navigating to home page

@app.route('/predict',methods=['POST'])
def upload():
    if request.method=='POST':
        img=request.files['file']                          # image upload
        dir_name=os.path.dirname(__file__)                 # current directory
        file_path=os.path.join(dir_name,'upload',secure_filename(img.filename))    # generating path

        img.save(file_path)                                # saving image to the path
        p=Predict()                                        # initializing the prediction class
        pred = p.model_predict(file_path, model)           # prediction

        return pred
    else:
        return None


if __name__=='__main__':
    app.run(debug=True)                                     # our application starts from here