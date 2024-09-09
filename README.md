Machine Learning Model Training and Testing System
Overview
This project is designed to implement machine learning algorithms from scratch, focusing on both regression and classification tasks. It provides functionality to train models such as linear regression, polynomial regression, support vector machines (SVM), and decision trees from the ground up, without relying on built-in model implementations from libraries like scikit-learn.

The project also includes preprocessing steps for input data and offers metrics to evaluate the performance of each model after training, such as mean squared error, R-squared, accuracy, and confusion matrices.

Key Features
Training Machine Learning Models from Scratch:
Linear Regression
Polynomial Regression
Support Vector Machine (SVM)
Decision Tree Classifier
Dataset Preprocessing:
The system includes preprocessing functions to prepare datasets for regression and classification models.
Features such as Subtitle, In-app Purchases, Number of Words, Size, Original Release Year, and Current Version Release Year are used to predict outputs like Average User Rating.
Performance Metrics:
Mean Squared Error (MSE): Evaluates the accuracy of regression models.
R-squared (R²): Measures how well the independent variables explain the variance in the dependent variable for regression models.
Confusion Matrix: Evaluates the performance of classification models.
Accuracy: Measures the percentage of correct predictions for classification models.
How It Works
Dataset Input:
The user provides a dataset in CSV format, which is used for both training and testing the models.
Preprocessing:
Data is preprocessed based on the task (regression or classification) using a custom preprocessing module (Pre_processing), which handles data cleaning, feature selection, and transformation.
Model Training:
Models like linear and polynomial regression, SVM, and decision trees are trained from scratch, using mathematical formulations rather than pre-built libraries.
Model Testing:
After training, the models are tested on unseen data, with various metrics being calculated to evaluate their performance.
User Interaction:
The user is prompted to choose between different models (e.g., linear or polynomial regression, SVM, decision tree), and test them with provided data.
Results Display:
Metrics such as MSE, R², accuracy, and confusion matrices are displayed to give insight into the model's performance on the test data.
Example of Usage
The user is prompted to input the name of the CSV file containing the dataset.
The user selects whether they want to train and test regression or classification models.
The script preprocesses the dataset and trains the chosen model from scratch.
After training, the system tests the model on test data and outputs performance metrics.


Sample Workflow:
$ python main.py
Enter the name of your dataset (e.g., data.csv).
Choose whether you want to train a regression or classification model.
Select the specific model you want to train (e.g., linear regression, polynomial regression, SVM).
View the evaluation results for the model after testing.
Dependencies
Pandas: For handling and manipulating datasets.
scikit-learn (metrics only): For evaluating model performance.
joblib: For saving and loading trained models.
Preprocessing module (Pre_processing): Custom module to preprocess the dataset for training and testing.
How to Run the Project
Clone the repository to your local machine.
Install the required dependencies using:
pip install -r requirements.txt
Run the main script and follow the prompts to input the dataset file, choose the model type, and start the training and testing process.
$ python main.py
After training, performance metrics will be displayed based on the chosen model and the test dataset.
Future Enhancements
Implementation of additional machine learning models, such as Random Forest and Neural Networks, from scratch.
Further optimization of the training algorithms for larger datasets.
Integration of cross-validation techniques to improve model generalization.
This description clearly explains that the models are implemented and trained from scratch, and it outlines the detailed steps and functionality provided by the project.
