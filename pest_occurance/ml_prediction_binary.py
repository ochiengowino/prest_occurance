import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression



# read the bunary data as a data frame
df= pd.read_csv('sample_data/Farmer_Tuta Absoluta_Occurence_Data.csv', encoding='unicode_escape')

bin_data = df

# drop names columns as they are not needed in the model
bin_data = bin_data.drop(['Name_1','Name_2'], axis=1)

# drop the column that contains string values
bin_data = bin_data.select_dtypes(exclude=['object'])

# select the train and the test values
X = bin_data.iloc[:, 1:]
y = bin_data['5_Aug_17']

# separate the train and test data
x_Train, x_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=100, shuffle=True)

# use Logistic Regression since there is binary data
log = LogisticRegression()

log.fit(x_Train, y_Train)

# predict the outcome using the test data
predictor = log.predict(x_Test)

# obtain the test score
score_bin = log.score(x_Test,y_Test)

# combine the test and the predicted outcome data
combined_dat = pd.concat([x_Test, pd.Series(predictor.flatten())], axis=1)

# rename the column containing the predicted data
combined_dat.rename(columns = {0:'PREDICTED'}, inplace = True)

combined_dat.to_csv('sample_data/predicted_data_bin.csv')