Diabetes Prediction Model
Overview
This project aims to develop a machine learning model to predict the likelihood of diabetes in individuals based on various health metrics. The model is trained on a dataset containing information about pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age of individuals.

Dataset
The dataset used for training the model is sourced from diabetes.csv. It contains the following columns:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome (0 for non-diabetic, 1 for diabetic)
Preprocessing
Handling missing values: Checked for missing values in the dataset and replaced them with mean values.
Dealing with zero values: Replaced zero values in BMI, blood pressure, glucose, insulin, and skin thickness columns with their respective means.
Handling outliers: Removed outliers from the dataset using quantile-based methods.
Scaling: Standardized the features using StandardScaler to bring them to the same scale.
Model Building
Used logistic regression to build the prediction model.
Split the dataset into training and testing sets with a 75-25 split.
Achieved a training accuracy of 77.08% and an adjusted R-squared score of 76.76%.
Model Evaluation
Evaluated the model on the test set and achieved an accuracy of 79.69%.
Calculated precision, recall, F1-score, and AUC-ROC score for model performance evaluation.
Plotted the ROC curve to visualize the model's performance.
Model Persistence
Saved the trained logistic regression model using pickle for future use in prediction.
Files Included
diabetes.csv: Dataset used for training the model.
standardScalar.pkl: Saved scaler object for feature scaling.
modelForPrediction.pkl: Saved logistic regression model for diabetes prediction.
README.md: Documentation for the project.
Requirements
Python 3.x
Libraries: pandas, numpy, scikit-learn, statsmodels, seaborn, matplotlib
Usage
Clone the repository to your local machine.
Install the required libraries using pip install -r requirements.txt.
Run the Jupyter Notebook or Python script to train the model and perform predictions.
Follow the instructions provided in the notebook/script for data preprocessing, model building, and evaluation.
