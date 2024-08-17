import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split

data=pd.read_excel("Book2.xlsx")

x_data=np.array(data.fuel_rate)
y_data=np.array(data.co2)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)


regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)



# # yhat = regr.coef_[0][0] * X_test + regr.intercept_[0]
yhat=regr.predict(X_test)
# y_=regr.predict(x_data)

mse=mean_squared_error(y_test,yhat)
mae=mean_absolute_error(y_test,yhat)
r2=r2_score(y_test,yhat)
print("mse",mse)
print("mae",mae)
print("r2",r2)

# plt.scatter(X_train,y_train,color="blue")
# plt.plot(x_data,y_)
# plt.scatter(X_test,y_test,color="red")
# plt.xlabel("fuel rate")
# plt.ylabel("CO2")
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')


# # plotting
# ax.plot3D(x_data[:,0], x_data[:,1], y_data, 'green')

# plt.show()

