import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# Load the binary data
# df= pd.read_csv('sample_data/Farmer_Tuta Absoluta_Occurence_Data.csv', encoding='unicode_escape')

# df.head()

# Load the temperature data
df3 = pd.read_csv('sample_data/Landsat 8-Land surface emperature_KathaanaFarms.csv', encoding='unicode_escape')


# drop the redundant columns
data = df3.drop(['FID_1','First_Name','SecondName', 'pointid', 'ID_No', 'Telephone'], axis=1)

# replacing #NUM! with zero values
data = data.replace('#NUM!',0)

# convert the whole df to numerical
data = data.astype(float)

# drop columns with null values
# data = data.dropna(axis=1)


# separating the training and testing data using the traing-test split method
train_data = data.iloc[:, 0:100]
test_data = data.iloc[:,101:]

x_train, x_test, y_train, y_test = train_test_split(train_data, test_data, test_size=0.2)

# Using Linear Regression as the classifier
regressor = LinearRegression()

regressor.fit(x_train, y_train)


predict = regressor.predict(x_test)


# obtaining the test score
score = regressor.score(x_test, y_test)


# combine the test and the predicted outcome
combined = pd.concat([x_test, pd.Series(predict.flatten())], axis=1)

combined=combined.dropna(how='all').reset_index(drop=True)

# renaming the column with the predicted values
combined.rename(columns = {0:'PREDICTED'}, inplace = True)

# export the final output as csv
combined.to_csv('sample_data/predicted_data.csv')