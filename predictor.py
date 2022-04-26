import sys
import pickle
import datetime
import pandas as pd

# load it back in.
with open('/mnt/code/my_linear_regression.sav', 'rb') as pickle_file:
     regression_model_2 = pickle.load(pickle_file)

# make a new prediction.
def predict(value):
    prediction = regression_model_2.predict([[value]])
    predicted_value = prediction[0][0]
    print("For a barrel of oil that costs - ", value, ", the predicted value of stock price is - , " , predicted_value)

#arg = sys.argv[1]
#predict(float(arg))

for arg in sys.argv[1:]:
  predict(float(arg))