# Credit-card-fraud-prediction
This Credit Card Fraud Detection project seeks to accurately identify fraudulent transactions in a highly skewed dataset in which only a small percentage of transactions are fraudulent. The dataset contains approximately 284,000 credit card transactions with anonymised attributes, enabling for detailed research without privacy concerns.
The project includes numerous important procedures for preprocessing, analyzing, and modeling data in order to efficiently detect fraud. Data cleansing, feature scaling, and class imbalance resolution using techniques such as SMOTE and class weight adjustments are all critical aspects. A variety of models will be evaluated, with decision trees. Performance will be measured using measures such as precision, recall, and F1-score, with an emphasis on reducing false positives while increasing fraud detection accuracy.
Available on Kaggle: Credit Card Fraud Detection Dataset.
Key Steps
Data Preprocessing:

Scaled the Time and Amount columns using RobustScaler.
Handled class imbalance through undersampling and oversampling techniques (SMOTE).
Removed duplicates and checked for missing values.
Exploratory Data Analysis:

Visualized transaction distributions and class imbalance.
Analyzed correlations between features using heatmaps.
Modeling:

Tested multiple machine learning algorithms:
Logistic Regression
Random Forest Classifier
Decision Tree Classifier
XGBoost Classifier
Evaluated models using metrics like precision, recall, F1-score, and accuracy.
Results:

Achieved over 99% accuracy using Random Forest and XGBoost on oversampled data.
Balanced precision and recall to minimize false positives.
Code Example
Data Loading and Exploration
import pandas as pd

data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print(data.head())
Model Evaluation
from sklearn.metrics import classification_report

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name}:\n{classification_report(y_test, y_pred)}")
Deployment
Saved the final model as a .pkl file using joblib for deployment.
Example usage:
import joblib

model = joblib.load('Credit_card_model.pkl')
pred = model.predict(new_data_point)
print("Fraud" if pred[0] == 1 else "No Fraud")
Technologies Used
Python
Pandas, NumPy
Scikit-learn
XGBoost
Seaborn, Matplotlib
Results Visualization
Class Distribution Correlation Matrix

Future Improvements
Explore deep learning models for enhanced fraud detection.
Implement real-time fraud detection using streaming frameworks.
