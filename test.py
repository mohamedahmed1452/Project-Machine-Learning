import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
import Pre_processing as pp
import warnings

warnings.filterwarnings('ignore')


def apply_test_script(path_csv_file, ML):
    if ML == "regression":
        # read dataset
        data = pd.read_csv(path_csv_file)
        print(len(data))
        data = pp.preprocessing(data, 'regression', 1)
        print(len(data))
        X_Test = data[
            ['Subtitle', 'In-app Purchases', 'no# of words', 'size_Q1', 'size_Q3', 'size_Q4', 'Original Release Year',
             'Current Version Release Year']]

        Y_Test = data['Average User Rating'].values
        while True:
            print(
                "Enter 1 to apply test on the model linear regression OR Enter 2 to apply test on the model "
                "polynomial regression")
            choice = input()
            if choice == '1':
                reg_loaded = joblib.load('linear_reg.joblib')
                Y_pred = reg_loaded.predict(X_Test)

                print("MSE data test script  to model linear regression :", metrics.mean_squared_error(Y_pred, Y_Test))

                # print the accuracy of the model
                print("Accuracy data test script  to model linear regression: {:.2f}%".format(
                    r2_score(Y_Test, Y_pred) * 100))

            elif choice == '2':
                Poly_reg_loaded = joblib.load('Poly_reg.joblib')
                # Transform the test data using the same polynomial features object used to train the model
                poly_features = PolynomialFeatures(degree=2)
                X_Test_poly = poly_features.fit_transform(X_Test)

                # Predict the output values for the test data
                y_pred = Poly_reg_loaded.predict(X_Test_poly)

                # Calculate the mean squared error and accuracy of the model
                # Print the results
                print("MSE data test script to model polynomial regression:",
                      metrics.mean_squared_error(Y_Test, y_pred))
                print("Accuracy data test script  to model polynomial regression: {:.2f}%".format(
                    r2_score(Y_Test, y_pred) * 100))

                print("#" * 100)
            print("Do you want to continue in apply testing on Regression Model y/n")
            ch = input()
            if ch == 'y':
                continue
            else:
                break
    elif ML == "classify":

        # read dataset
        data = pd.read_csv(path_csv_file)
        data = pp.preprocessing(data, 'classify', 1)
        print(len(data))
        X_Test = data[['Subtitle', 'In-app Purchases', 'size_Q1', 'size_Q3', 'size_Q4', 'Original Release Year',
                       'Current Version Release Year']]

        Y_Test = data['Rate'].values
        while True:
            print(
                "Enter 1 to apply test on the  model logistic regression OR Enter 2 to apply test on the model SVM OR "
                "Enter 3 to apply test on the model Decision tree")
            choice = input()
            if choice == '1':
                logistic_reg_test = joblib.load('logistic_reg.joblib')
                Y_pred_log = logistic_reg_test.predict(X_Test)
                # Print the confusion matrix and classification report
                print('confusion_matrix for testing data:')
                print(confusion_matrix(Y_Test, Y_pred_log))

                accuracy = accuracy_score(Y_Test, Y_pred_log)
                print("Accuracy for testing data model logistic regression: {:.2f}%".format(accuracy * 100))

            elif choice == '2':
                classifier_svm_test = joblib.load('svm_model.joblib')

                # Predict target labels for training data
                y_pred_svm = classifier_svm_test.predict(X_Test)

                # Print confusion matrix for training data
                print('Confusion matrix for testing data model svm:')
                print(confusion_matrix(Y_Test, y_pred_svm))

                # Evaluate the performance of the model
                accuracy = accuracy_score(Y_Test, y_pred_svm)
                print("Accuracy for testing data model svm: {:.2f}%".format(accuracy * 100))

            elif choice == '3':
                classifier_DT_test = joblib.load('decision_tree.joblib')

                # Predict target labels for training data
                y_pred_DT = classifier_DT_test.predict(X_Test)

                # Print confusion matrix for training data
                print('Confusion matrix for testing data model Decision tree:')
                print(confusion_matrix(Y_Test, y_pred_DT))

                # Evaluate the performance of the model
                accuracy = accuracy_score(Y_Test, y_pred_DT)
                print(len(y_pred_DT))
                print("Accuracy for testing data model Decision tree: {:.2f}%".format(accuracy * 100))

            print("Do you want to continue in apply testing on Classification Model y/n")
            ch = input()
            if ch == 'y':
                continue
            else:
                break


while True:
    print(" If you want test Regression Models Enter 1 "
          " and If you want  test Classification Models enter 2")
    x = input()
    if x == '1':
        print("Enter name's csv File for example file_name.csv ")
        file_name = input()
        apply_test_script(file_name, 'regression')
    elif x == '2':
        print("Enter name's csv File for example file_name.csv ")
        file_name = input()
        apply_test_script(file_name, 'classify')
    else:
        print("this choice is wrong please try again")
        continue
    print("Do you want to continue in this Program y/n")
    contin = input()
    if contin == 'Y' or contin == 'y':
        continue
    else:
        break

# games-regression-dataset.csv


# games-classification-dataset.csv



#ms2-games-tas-test-v2.csv
#ms1-games-tas-test-v2.csv