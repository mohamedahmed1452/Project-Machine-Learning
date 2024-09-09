import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import Pre_processing as pp
import joblib
import warnings

warnings.filterwarnings('ignore')
# read dataset
all_data = pd.read_csv("games-regression-dataset.csv")
# split data into data train and data test
data_train_for_all_dataset, data_test = train_test_split(all_data, test_size=0.3, random_state=0)

data_train_for_all_dataset = pd.DataFrame(data_train_for_all_dataset)
data_test = pd.DataFrame(data_test)

data = pp.preprocessing(data_train_for_all_dataset, 'regression', 0)

#
# X1 = data['Subtitle']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('Subtitle', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['In-app Purchases']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('In-app Purchases', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['size_Q1']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('size_Q1', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['size_Q2']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('size_Q2', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['size_Q3']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('size_Q3', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['size_Q4']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('size_Q4', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['Price']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('Price', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['no# of words']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('no# of words', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['Original Release Year']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('Original Release Year', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()
#
# X1 = data['Current Version Release Year']
# Y = data['Average User Rating']
# cls = linear_model.LinearRegression()
# X1 = np.expand_dims(X1, axis=1)
# Y = np.expand_dims(Y, axis=1)
# cls.fit(X1, Y)  # Fit method is used for fitting your training data into the model
# prediction = cls.predict(X1)
# plt.scatter(X1, Y)
# plt.xlabel('Current Version Release Year', fontsize=20)
# plt.ylabel('Average User Rating', fontsize=20)
# plt.plot(X1, prediction, color='red', linewidth=3)
# plt.show()


#######################################################################################################
#######################################################################################################
corr = data.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Average User Rating']) > 0.07]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

top_feature = list(top_feature)
top_feature.remove('Average User Rating')

y = data['Average User Rating'].values
x = data[top_feature]
# # Split data: 70% training set, 30% testing set

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=1 / 3, random_state=0)
regressor = LinearRegression()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_valid)  # validation

# plot
plt.scatter(y_valid, y_pred)
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], linestyle='dashed')
plt.title('Linear Regression Model')
plt.xlabel('Actual Data')
plt.ylabel('Prediction Data')
plt.show()

print("MSE data train  to model linear regression :", metrics.mean_squared_error(regressor.predict(x_train), y_train))

print("MSE data valid  to model linear regression :", metrics.mean_squared_error(y_pred, y_valid))

print("Accuracy data train  to model linear regression: {:.2f}%".format(
    metrics.r2_score(y_train, regressor.predict(x_train)) * 100))
print("Accuracy data valid  to model linear regression: {:.2f}%".format(metrics.r2_score(y_valid, y_pred) * 100))

print("#" * 100)
# score of degree 2


# Save model
joblib.dump(regressor, 'linear_reg.joblib')
poly_features = PolynomialFeatures(degree=2)  # ###########################################################

# Transform the input features X into polynomial features
X_poly = poly_features.fit_transform(x_train)

# Initialize a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_poly, y_train)

# Create a new set of input features X_test
X_test = x_valid

# Transform the test data into polynomial features using the same poly_features object
X_test_poly = poly_features.transform(X_test)

# Use the trained model to predict the output values for the test data
Y_pred = model.predict(X_test_poly)

# plt.scatter(x_train, y_train, color='red')
# plt.plot(x_train, y_pred, color='blue')
# plt.title('polynomial regression Model')
# plt.xlabel('Actual Data')
# plt.ylabel('Prediction Data')
# plt.show()

# plot
plt.scatter(y_valid, Y_pred)
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], linestyle='dashed')
plt.title('polynomial regression Model')
plt.xlabel('Actual Data')
plt.ylabel('Prediction Data')
plt.show()
# plt.plot(X_test_poly, y_pred)
# plt.scatter(x_valid, y_valid)
# plt.show()

print("MSE data train  to model polynomial regression :",
      metrics.mean_squared_error(model.predict(poly_features.transform(x_train)), y_train))
print("MSE data valid  to model polynomial regression :", metrics.mean_squared_error(Y_pred, y_valid))
accuracy = r2_score(Y_pred, y_valid)
# print the accuracy of the model
print("Accuracy data train  to model polynomial regression: {:.2f}%".format(
    metrics.r2_score(y_train, model.predict(poly_features.transform(x_train))) * 100))
print("Accuracy data valid  to model polynomial regression: {:.2f}%".format(
    metrics.r2_score(y_valid, model.predict(poly_features.transform(x_valid))) * 100))

print("#" * 100)

# Save the trained model to a file
# with open('Polynomial Regression model.pkl', 'wb') as file:
#     pickle.dump(model, file)
# Save model
joblib.dump(model, 'Poly_reg.joblib')
# #######################################################################################################
# #######################################################################################################
data_test = pp.preprocessing(data_test, 'regression', 0)

# #######################################################################################################
# #######################################################################################################


Y_test = data_test['Average User Rating'].values
X_test = data_test[top_feature]

reg_loaded = joblib.load('linear_reg.joblib')
Y_pred_test = reg_loaded.predict(X_test)
# Y_pred_test = model1.predict(X_test)  # testing
print("MSE data test  to model linear regression:", metrics.mean_squared_error(Y_pred_test, Y_test))
# calculate the accuracy of the model using r2_score
# print the accuracy of the model
print("Accuracy data test  to model linear regression: {:.2f}%".format(r2_score(Y_test, Y_pred_test) * 100))

print("#" * 100)
# regress.fit(X_poly,Y_test)


# Create a new set of input features X_test
# X_test = X_test
# #############################################################################################################
# Load the saved model back into memory
Poly_reg_loaded = joblib.load('Poly_reg.joblib')

# Transform the test data into polynomial features using the same poly_features object
X_test_poly = poly_features.transform(X_test)

# Use the trained model to predict the output values for the test data
y_pred_test = Poly_reg_loaded.predict(X_test_poly)
print("MSE data test  to model polynomial regression:", metrics.mean_squared_error(y_pred_test, Y_test))

# print the accuracy of the model
print("Accuracy data test  to model polynomial regression: {:.2f}%".format(r2_score(Y_test, y_pred_test) * 100))

print("#" * 100)
