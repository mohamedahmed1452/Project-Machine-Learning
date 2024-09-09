import time
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import Pre_processing as pp
from sklearn.metrics import accuracy_score
from sklearn import svm
import warnings

warnings.filterwarnings('ignore')
# read dataset
all_data = pd.read_csv("games-classification-dataset.csv")
# split data into data train and data test
data_train_for_all_dataset, data_test = train_test_split(all_data, test_size=0.2, random_state=0)

data_train_for_all_dataset = pd.DataFrame(data_train_for_all_dataset)
data_test = pd.DataFrame(data_test)

data = pp.preprocessing(data_train_for_all_dataset, 'classify', 0)

corr = data.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.07]

# print(top_feature)
top_feature = list(top_feature)
top_feature.remove('Rate')
y = data['Rate'].values
x = data[top_feature]

# Logistic regression training

start_train_log_time = time.time()
#solver='lbfgs', max_iter=0 underfitting
#solver='lbfgs', max_iter=1000 best fit
#solver='lbfgs', max_iter=100000; overfitting
classifier_log = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
classifier_log.fit(x, y)
y_pred_train_log = classifier_log.predict(x)
finish_train_log_time = time.time() - start_train_log_time

print('confusion_matrix for training data model logistic regression:')
print(confusion_matrix(y, y_pred_train_log))
# Evaluate the performance of the model
accuracy = accuracy_score(y, y_pred_train_log)
print("Accuracy for training data model logistic regression: {:.2f}%".format(accuracy * 100))
# Save model
joblib.dump(classifier_log, 'logistic_reg.joblib')
#####################################################################################
#####################################################################################

# SVM training

# Initialize SVM classifier with RBF kernel and C=1, gamma=1
# the best parameter at c=100 and kernel='poly' acc=0.5708812260536399 over fitting
# the best parameter at c=0.3 and kernel='poly' acc=0.6398467432950191  best fit
# the best parameter at c=0.0009 and kernel='poly' acc=0.49808429118773945  under fitting
# the best parameter at c=0.3 and kernel='rbf' acc=63.98%
# the best parameter at c=0.3 and kernel='linear' acc=62.45%
# the best parameter at c=0.3 and kernel='poly', degree = 3 acc = 49.81%   under fitting
# the best parameter at c=0.25 and kernel='rbf', gamma = 0.5 acc = 63.22%   best fit
# 0.5952305246422893

start_train_svm_time = time.time()
classifier_svm = svm.SVC(kernel='rbf', C=0.25, gamma=0.5)  # gamma = 1 / (2 * sigma^2).

# Fit the model to training data
classifier_svm.fit(x, y)

# Predict target labels for training data
y_pred_train_svm = classifier_svm.predict(x)
finish_train_svm_time = time.time() - start_train_svm_time

# Print confusion matrix for training data
print('Confusion matrix for training data model svm:')
print(confusion_matrix(y, y_pred_train_svm))

# Evaluate the performance of the model
accuracy = accuracy_score(y, y_pred_train_svm)
print("Accuracy for training data model svm: {:.2f}%".format(accuracy * 100))

# Print the value of C and gamma used in the model
print(f"C value used in the model SVM: {classifier_svm.C}")
# Save the model to a file
joblib.dump(classifier_svm, 'svm_model.joblib')

#####################################################################################
#####################################################################################

# Decision tree training
start_train_DT_time = time.time()
classifier_DT = DecisionTreeClassifier(max_depth=4)

# Fit the model to training data
classifier_DT.fit(x, y)

# Predict target labels for training data
y_pred_train_dt = classifier_DT.predict(x)
finish_train_DT_time = time.time() - start_train_DT_time

# Print confusion matrix for training data
print('Confusion matrix for training data model Decision tree:')
print(confusion_matrix(y, y_pred_train_dt))

# Evaluate the performance of the model
accuracy = accuracy_score(y, y_pred_train_dt)
print("Accuracy for training data model Decision tree: {:.2f}%".format(accuracy * 100))

# Save model
joblib.dump(classifier_DT, 'decision_tree.joblib')
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

data_test = pp.preprocessing(data_test, 'classify', 0)

Y_test = data_test['Rate'].values
X_test = data_test[top_feature]

# Logistic regression testing

# Load the saved model logistic_reg
start_test_log_time = time.time()

logistic_reg_loaded = joblib.load('logistic_reg.joblib')
y_pred_test_log = logistic_reg_loaded.predict(X_test)
finish_test_log_time = time.time() - start_test_log_time

# Print the confusion matrix and classification report
print('confusion_matrix for testing data model logistic regression :')
print(confusion_matrix(Y_test, y_pred_test_log))

accuracy_log = accuracy_score(Y_test, y_pred_test_log)
print("Accuracy for testing data model logistic regression: {:.2f}%".format(accuracy_log * 100))

#####################################################################################
#####################################################################################
# svm testing

# Load the saved model svm_model
start_test_svm_time = time.time()
classifier_svm_loaded = joblib.load('svm_model.joblib')

# Predict target labels for training data
y_pred_test_svm = classifier_svm_loaded.predict(X_test)
finish_test_svm_time = time.time() - start_test_svm_time
# Print confusion matrix for training data
print('Confusion matrix for testing data model svm:')
print(confusion_matrix(Y_test, y_pred_test_svm))

# Evaluate the performance of the model
accuracy_svm = accuracy_score(Y_test, y_pred_test_svm)
print("Accuracy for testing data model svm: {:.2f}%".format(accuracy_svm * 100))

#####################################################################################
#####################################################################################
# Decision tree testing

# Load the saved model svm_model
start_test_DT_time = time.time()

classifier_DT_loaded = joblib.load('decision_tree.joblib')

# Predict target labels for training data
y_pred_test_DT = classifier_DT_loaded.predict(X_test)
finish_test_DT_time = time.time() - start_test_DT_time

# Print confusion matrix for training data
print('Confusion matrix for testing data model Decision tree:')
print(confusion_matrix(Y_test, y_pred_test_DT))

# Evaluate the performance of the model
accuracy_DT = accuracy_score(Y_test, y_pred_test_DT)
print("Accuracy for testing data model Decision tree: {:.2f}%".format(accuracy_DT * 100))


# Define the data
accuracy = [accuracy_log, accuracy_svm, accuracy_DT]  # classification accuracy for three models
training_time = [finish_train_log_time, finish_train_svm_time,
                 finish_train_DT_time]  # total training time for three models in seconds
test_time = [finish_test_log_time, finish_test_svm_time,
             finish_test_DT_time]  # total test time for three models in seconds

# Define the x-axis labels
labels = ['logistic', 'svm', 'Decision tree']

# Define the x-axis locations
x = np.arange(len(labels))

# Create the figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot the classification accuracy
axs[0].bar(x, accuracy)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].set_ylim([0, 1])
axs[0].set_title('Classification Accuracy')

# Plot the total training time
axs[1].bar(x, training_time)
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].set_ylim([0, max(training_time) * 1.2])
axs[1].set_title('Total Training Time (seconds)')

# Plot the total test time
axs[2].bar(x, test_time)
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels)
axs[2].set_ylim([0, max(test_time) * 1.2])
axs[2].set_title('Total Test Time (seconds)')

# Add labels to the y-axis
for ax in axs:
    ax.set_ylabel('Value')

# Show the plot
plt.show()
