from flask import Flask, request, render_template
from flask_cors import cross_origin
#import sklearn
#import pickle
import pandas as pd
import numpy as np

import joblib as joblib
import requests


app = Flask(__name__)
#model = pickle.load(open("model_path.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

       
       
        # R&D_Expenses
        RD_Expenses = float(request.form["R&D_Expenses"])
        # print(RD_Expenses)


        # Admin_Expenses
        Admin_Expenses = float(request.form["Admin_Expenses"])
        # print(Admin_Expenses)


        # Marketing_Expenses
        Marketing_Expenses = float(request.form["Marketing_Expenses"])
        # print(Marketing_Expenses)


        # State
        state=request.form['state']

        data = [RD_Expenses, Admin_Expenses, Marketing_Expenses, state]
    
        X = [RD_Expenses, Admin_Expenses, Marketing_Expenses, state]
        X = pd.DataFrame([X])
        X = X.values
        #print(X)



        # load encoder
        labelencoder = joblib.load('labelEncoder.joblib')

        X[:, 3] = labelencoder.transform(X[:, 3])
        
        #print(X)




        # load it when test
        ct = joblib.load('oneHotEncoder.joblib')
        X = ct.transform(X)
        #print(X)


        # Avoiding the Dummy Variable Trap
        X = X[:, 1:]
        print(X)


        #load model file locally and get prediction
        #model = joblib.load(open("model.joblib", "rb"))
        #prediction=model.predict(X)
        #output=round(prediction[0],2)
        #print(output)



        #get prediction from model endpoint deployed on aws as api ( stored in heroku config)
        import os
        model_api = os.getenv("AWS_MODEL_API_URL")
        
        r = requests.post(model_api, json={
        "data":X.tolist()
        })
        
        output = round(r.json(),2)
        print(output)


        return render_template('results.html',prediction_text="Startup's Predicted Profit is Rs. {}".format(output))

    
    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
