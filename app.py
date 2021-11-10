from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import os

app=Flask(__name__)
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('user uploads' , filename)
        file.save(file_path)

        model = load_model("breast_cancer_classifier.h5")
        img= image.load_img(file_path, target_size=(32,32))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        result = int(classes[0][0])

        if result==0:
            return render_template('predict.html', prediction='The tumor is benign..Need not worry!')
        else:
            return render_template('predict.html', prediction='It is a malignant tumor...Please Consult Doctor')


if __name__ == "__main__":
    app.run(debug =True)
