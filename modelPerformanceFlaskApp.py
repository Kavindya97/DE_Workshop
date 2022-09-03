from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import pickle
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__,template_folder='template')

def print_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Accuracy score:\n', metrics.accuracy_score(y_test, y_pred))
    print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))
    print('Classification report:\n', metrics.classification_report(y_test, y_pred, digits=4))
    print('Precision:\n',metrics.precision_score(y_test, y_pred))
    print('Recall:\n',metrics.recall_score(y_test, y_pred))


def dataSplit():
    psb_data = pd.read_csv('phosphate_solubilizing_bacteria.csv')
    X = psb_data.drop('Viability', axis=1) #indep. features
    y = psb_data['Viability'] #dep. feature
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
    

@app.route('/predict', methods=['GET', 'POST'])
def model_predict():
    #dataSplit()
    psb_data = pd.read_csv('phosphate_solubilizing_bacteria.csv')
    psb_data.drop('Label no', axis=1, inplace=True)
    X = psb_data.drop('Viability', axis=1) #indep. features
    y = psb_data['Viability'] #dep. feature
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
    
    if request.method == 'GET':
        #Pickle
        print('\n--- Pickle ---- ')
        loaded_model = pickle.load(open('psb_lr_model.pkl', 'rb'))
        print_model_performance(loaded_model, X_test, y_test)

        #Joblib
        print('\n--- Joblib ---- ')
        loaded_model = joblib.load('psb_lr_model.jblb')
        print_model_performance(loaded_model, X_test, y_test)
        
    return render_template('predict.html')



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
