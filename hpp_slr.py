"""  This is my first attempt on a simple linear regression model to predict housing prices.
 Still working on improving the accuracy of the model though. This is very simple code with not much functionality. """

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

# importing dataset
db= pd.read_csv(r"C:\Users\SHRUTHI\OneDrive\Desktop\FSDS\Assignments\projects\projects to do\pj-12 hpp, slr&mlr\SLR - House price prediction\House_data.csv")
space=db['sqft_living']
price=db['price']

x=np.array(space).reshape(-1, 1)
y=np.array(price)

#Splitting the data into train and test set

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3, random_state=0)

# fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(xtrain, ytrain)

# Predicting the prices
pred= regressor.predict(xtest)

# Visualizing the training test results
plt.scatter(xtrain,ytrain, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('Visuals of training Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

# Visualizing the test results
plt.scatter(xtest,ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title('Visuals of test Dataset')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()