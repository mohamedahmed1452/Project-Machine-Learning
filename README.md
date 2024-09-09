
# Machine Learning Model Training and Testing System

## Overview
This project implements machine learning algorithms from scratch for both regression and classification tasks. It includes training models like **linear regression**, **polynomial regression**, **support vector machines (SVM)**, and **decision trees** without using built-in libraries like scikit-learn.

The project also handles data preprocessing and evaluates model performance using metrics such as **Mean Squared Error (MSE)**, **R-squared**, **accuracy**, and **confusion matrices**.

## Key Features
- **Models Trained from Scratch**:
  - Linear Regression
  - Polynomial Regression
  - Support Vector Machine (SVM)
  - Decision Tree Classifier
- **Dataset Preprocessing**: Prepares datasets for regression and classification tasks, using features like `Subtitle`, `In-app Purchases`, `Size`, and `Release Year`.
- **Performance Evaluation**:
  - Mean Squared Error (MSE) for regression accuracy
  - R-squared (R²) for variance explanation in regression
  - Confusion Matrix and Accuracy for classification performance

## How It Works
1. **Dataset Input**: Users provide a CSV dataset for training and testing.
2. **Preprocessing**: Custom preprocessing handles data cleaning, feature selection, and transformation.
3. **Model Training**: Models are trained from scratch using mathematical formulations.
4. **Model Testing**: After training, models are tested on unseen data, and performance is evaluated using various metrics.
5. **Results Display**: Performance metrics (MSE, R², accuracy, confusion matrix) are shown after testing.

## Example Usage
1. Enter the dataset name when prompted.
2. Choose a model type (regression or classification).
3. Select a specific model to train (e.g., SVM, Decision Tree).
4. View performance metrics after testing.

### Sample Commands:
```bash
$ python main.py
```

## Dependencies
- **Pandas**: For data manipulation.
- **scikit-learn (metrics)**: For performance evaluation.
- **joblib**: For saving/loading models.
- **Pre_processing module**: For custom data preprocessing.

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script and follow the prompts to input the dataset and model type:
   ```bash
   python main.py
   ```

## Future Enhancements
- Addition of models like Random Forest and Neural Networks.
- Optimization for larger datasets.
- Integration of cross-validation for better generalization.
