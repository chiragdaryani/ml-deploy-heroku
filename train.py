from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn import tree
#from sklearn.externals import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    

import joblib as joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    #parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    #args = parser.parse_args()

    #file = os.path.join(args.train, "50_Startups.csv")
    #dataset = pd.read_csv(file, engine="python")
    dataset = pd.read_csv("50_Startups.csv", engine="python")


    # labels are in the first column
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values


    print(X)

    # Encoding categorical data
    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])

    #print(X)



    #save label encoder
    # to save encoder 
    joblib.dump(labelencoder,'labelEncoder.joblib',compress=9)



    #onehotencoder = OneHotEncoder(categorical_features = [3])
    #X = onehotencoder.fit_transform(X).toarray()
    #print(X)


    #Encode Country Column
    ct = ColumnTransformer([("state encoder", OneHotEncoder(), [3])], remainder = 'passthrough')
    X = ct.fit_transform(X)
    #print(X)

    #save one hot encoder
    # to save encoder 
    joblib.dump(ct,'oneHotEncoder.joblib',compress=9)


    # Avoiding the Dummy Variable Trap
    X = X[:, 1:]

    #print(X)

    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    #joblib.dump(regressor, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(regressor, "model.joblib")



def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    #regressor = joblib.load(os.path.join(model_dir, "model.joblib"))
    regressor = joblib.load("model.joblib")
    return regressor
