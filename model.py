import sys
import pickle
import datetime
import pandas as pd
 
# load it back in.
with open('my_linear_regression.sav', 'rb') as pickle_file:
     regression_model_2 = pickle.load(pickle_file)
 
# make a new prediction.
def predict(value):
    prediction = regression_model_2.predict([[value]])
    predicted_value = prediction[0][0]
    print("Predicted value, " , predicted_value)
    return predicted_value