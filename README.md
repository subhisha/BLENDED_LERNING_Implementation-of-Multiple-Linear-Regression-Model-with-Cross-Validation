# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the car dataset (clean data, split into features X and target y).
2. Divide the dataset into K folds (e.g., K = 5).
3.Train the Multiple Linear Regression model on K−1 folds,Test on the remaining fold,Calculate error (MSE or R²). 
4. Compute the average error of all folds and train the final model on the full dataset for prediction.

## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
data= pd.read_csv('CarPrice_Assignment (1).csv')
data.head()
data = data.drop(['car_ID','CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
data.head()
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print('Name: SUBHISHA P')
print('Reg. No: 212225040143')
print("\n== cross-Validation ===")
cv_scores = cross_val_score(model, X, y, cv=5)
print("Fold R^2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R^2: {cv_scores.mean():.4f}")

y_pred = model.predict(X_test)
print(f"{'MAE':}: {mean_absolute_error(y_test,y_pred):}")

print("\n=== Test Set performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R^2: {r2_score(y_test, y_pred):.4f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted prices")
plt.grid(True)
plt.show()  
```

## Output:
<img width="1273" height="487" alt="image" src="https://github.com/user-attachments/assets/89ffdacd-bee1-45b5-8409-2c9a7fae1807" />
<img width="1255" height="345" alt="image" src="https://github.com/user-attachments/assets/932c2a48-881d-462f-a582-afd47a9df157" />
<img width="412" height="83" alt="image" src="https://github.com/user-attachments/assets/b7c1f49b-fcf7-44cd-8212-35ff7711dc44" />
<img width="857" height="560" alt="image" src="https://github.com/user-attachments/assets/24dcb519-65f1-4029-b7ff-4cd7a8f0ffa1" />
<img width="716" height="186" alt="image" src="https://github.com/user-attachments/assets/30dd7895-d659-4639-a709-81ea04857598" />
<img width="1191" height="697" alt="image" src="https://github.com/user-attachments/assets/c143111c-e8c5-4f1f-8120-4fea7b2a83db" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
