# Credit-card-fraud-prediction
This Credit Card Fraud Detection project seeks to accurately identify fraudulent transactions in a highly skewed dataset in which only a small percentage of transactions are fraudulent. The dataset contains approximately 284,000 credit card transactions with anonymised attributes, enabling for detailed research without privacy concerns.
The project includes numerous important procedures for preprocessing, analyzing, and modeling data in order to efficiently detect fraud. Data cleansing, feature scaling, and class imbalance resolution using techniques such as SMOTE and class weight adjustments are all critical aspects. A variety of models will be evaluated, with decision trees. Performance will be measured using measures such as precision, recall, and F1-score, with an emphasis on reducing false positives while increasing fraud detection accuracy.

## Dataset
- The dataset contains 284,807 transactions with 30 anonymized features (`V1` to `V28`), along with `Time`, `Amount`, and a binary `Class` label indicating fraud (1) or legitimate (0).
- Available on Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/).

## Key Steps
1. **Data Preprocessing**:
   - Scaled the `Time` and `Amount` columns using `standardscaler`.
   - Handled class imbalance through undersampling and oversampling techniques (SMOTE).
   - Removed duplicates and checked for missing values.

2. **Exploratory Data Analysis**:
   - Visualized transaction distributions and class imbalance.
   - Analyzed correlations between features using heatmaps.
   - used SMOTE to balance my class.

3. **Modeling**:
   - Tested multiple machine learning algorithms:
     - Logistic Regression
     - Random Forest Classifier
     - Decision Tree Classifier
     - XGBoost Classifier
   - Evaluated models using metrics like precision, recall, F1-score, and accuracy.

4. **Results**:
   - Achieved over 99% accuracy using Random Forest and XGBoost on oversampled data.
   - Balanced precision and recall to minimize false positives.

## Code Example
### Data Loading and Exploration
```python
import pandas as pd

data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print(data.head())
```

### Model Evaluation
```python
from sklearn.metrics import classification_report

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name}:\n{classification_report(y_test, y_pred)}")
```

## Deployment
- Saved the final model as a `.pkl` file using `joblib` for deployment.
- Example usage:
  ```python
  import joblib

  model = joblib.load('Credit_card_model.pkl')
  pred = model.predict(new_data_point)
  print("Fraud" if pred[0] == 1 else "No Fraud")
  ```
  ### Gradio-interface
  - Made the model accessible as a web app using gradio for real-time fraud detection!
    ```python
    import pandas as pd
    
    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import StandardScaler

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.metrics import classification_report

    import joblib

    import gradio as gr

   df = pd.read_csv('creditcard.csv')
   df.isnull().sum()
   df=df.dropna()
   df = df.drop_duplicates()
   X = df.drop(['Class'], axis=1)
   y = df['Class']
   scaler = StandardScaler()
   df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
   model = DecisionTreeClassifier(random_state=42)
   model.fit(X_train, y_train)
  joblib.dump(model, 'decision_tree_model.pkl')
  joblib.dump(scaler, 'scaler.pkl')
  predictions = model.predict(X_test)
  print(classification_report(y_test, predictions))
  def predict_fraud(*features):
 ### Convert input to DataFrame
   ```python
 features = [float(x) for x in features]
 features_df = pd.DataFrame([features], columns=X.columns)
 features_df[['Time', 'Amount']] = scaler.transform(features_df[['Time', 'Amount']])
 prediction = model.predict(features_df)
 return 'Fraud' if prediction[0] == 1 else 'Not Fraud'
feature_names = X.columns
interface = gr.Interface(
  fn=predict_fraud,
    inputs=[gr.Number(label=name) for name in feature_names],
    outputs="text",
    title="Credit Card Fraud Detection",
    description="Provide transaction features to predict if it's fraud or not."
)
interface.launch()
```

## Technologies Use
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Seaborn, Matplotlib

## Results Visualization
![Class Distribution](path/to/class-distribution-plot.png)
![Correlation Matrix](path/to/correlation-matrix.png)

## Future Improvements
- Explore deep learning models for enhanced fraud detection.
- Implement real-time fraud detection using streaming frameworks.

## Author
- **LinkedIn**: [VenkatasaireddyYerrappagari](linkedin.com/in/venkata-sai-reddy-yerrappagari-7304a32b7 )
- **GitHub**: [MonirulIslamm08](https://github.com/Venkatasaireddy984/workco.git)
- **Email**: venkatasaireddy.yerrappagari@gmail.com
