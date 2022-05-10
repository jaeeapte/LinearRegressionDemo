import pandas as pd
import numpy as np
import math 
import argparse
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from domino.data_sources import DataSourceClient
 
# instantiate a client and fetch the datasource instance
ds = DataSourceClient().get_datasource("UX-QUALITY")
 
# res is a simple wrapper of the query result
res = ds.query("select * from RAW_OIL_PRICE")
 
# to_pandas() loads the result into a pandas dataframe
df = res.to_pandas()
 
# define the new name and rename
new_column_names = {'EXON_PRICE':'EXXON_PRICE'}
df = df.rename(columns = new_column_names)
 
# drop any missing values
df = df.dropna()
 
#filter based on user input
parser = argparse.ArgumentParser(description='Filter on date')
parser.add_argument('num_days', type=int, nargs='?',
                    help='choose the number of days to filter data on, less than 1823', default = 1823)
args = parser.parse_args()
num_days = vars(args)["num_days"]
cutoff_day = df['DATE'].max() - pd.to_timedelta(num_days, unit='d')
price_data_filtered = df[df['DATE'] >= cutoff_day]
 
# Train
 
# define our input variable (X) & output variable.
Y = price_data_filtered.drop(['DATE','OIL_PRICE'], axis = 1)
X = price_data_filtered[['OIL_PRICE']]
 
# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
 
# create a Linear Regression model object.
regression_model = LinearRegression()
 
# pass through the X_train & y_train data set.
regression_model.fit(X_train, y_train)
 
# Get multiple predictions
y_predict = regression_model.predict(X_test)
 
# calculate the mean squared error, mean absolute error, and root mean squared error
model_mse = mean_squared_error(y_test, y_predict)
model_mae = mean_absolute_error(y_test, y_predict)
model_rmse =  math.sqrt(model_mse)
 
# calculate R2
model_r2 = r2_score(y_test, y_predict)
 
#display the output in Domino
import json
with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"Days Used for Training":num_days, "R^2": model_r2, "RMSE": model_rmse}))
 
 
print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))
print("R2: {:.2}".format(model_r2))