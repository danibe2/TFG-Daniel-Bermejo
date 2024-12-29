import pandas as pd                                       #importing pandas
import numpy as np                                        #importing numpy
import matplotlib.pyplot as plt                           #importing matplotlib 
import seaborn as sns                                     #importing seaborn
from sklearn.model_selection import train_test_split      #importing scikit-learn's function for data splitting
from sklearn.linear_model import LinearRegression         #importing scikit-learn's linear regressor function
from sklearn.neural_network import MLPRegressor           #importing scikit-learn's neural network function
from sklearn.ensemble import GradientBoostingRegressor    #importing scikit-learn's gradient booster regressor function
from sklearn.metrics import mean_squared_error            #importing scikit-learn's root mean squared error function for model evaluation
from sklearn.model_selection import cross_validate        #improting scikit-learn's cross validation function
import gurobipy as gp                                     #importing Gurobi
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
boxscores = pd.read_csv('base.csv')

# Acotación de la muestra
boxscores['fraud_score'] = (
    2 * boxscores['velocity_6h'] +
    1.5 * boxscores['velocity_24h'] -
    0.5 * boxscores['bank_months_count'] +
    3 * boxscores['proposed_credit_limit'] +
    1.25 * boxscores['credit_risk_score']
)

# Crear subplots para analizar variables
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.tight_layout()

# Gráficos de dispersión usando la columna 'fraud_score'
axes[0, 0].scatter(boxscores['name_email_similarity'], boxscores['fraud_score'], c='blue', alpha=0.2)
axes[0, 1].scatter(boxscores['velocity_6h'], boxscores['fraud_score'], c='green', alpha=0.2)
axes[0, 2].scatter(boxscores['customer_age'], boxscores['fraud_score'], c='orange', alpha=0.2)
axes[0, 3].scatter(boxscores['bank_months_count'], boxscores['fraud_score'], c='red', alpha=0.2)
axes[1, 0].scatter(boxscores['zip_count_4w'], boxscores['fraud_score'], c='purple', alpha=0.2)
axes[1, 1].scatter(boxscores['employment_status'], boxscores['fraud_score'], c='brown', alpha=0.2)
axes[1, 2].scatter(boxscores['credit_risk_score'], boxscores['fraud_score'], c='teal', alpha=0.2)
axes[1, 3].scatter(boxscores['device_os'], boxscores['fraud_score'], c='magenta', alpha=0.2)

# Etiquetas de los ejes
axes[0, 0].set_xlabel('Name Email Similarity')
axes[0, 1].set_xlabel('Velocity (6h)')
axes[0, 2].set_xlabel('Customer Age')
axes[0, 3].set_xlabel('Bank Months Count')
axes[1, 0].set_xlabel('Zip Count (4w)')
axes[1, 1].set_xlabel('Employment Status')
axes[1, 2].set_xlabel('Credit Risk Score')
axes[1, 3].set_xlabel('Device OS')

# Etiquetas del eje Y
for ax in axes.flat:
    ax.set_ylabel('Fraud Score')

# Mostrar gráficos
plt.show()

# Selección de variables (features) y target
X = boxscores[['velocity_6h', 'customer_age', 'bank_months_count', 'proposed_credit_limit', 'credit_risk_score', 'name_email_similarity', 'zip_count_4w']]  # características
y = boxscores['fraud_score']  # objetivo

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# 1. **Regresión Lineal**
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
linear_regression_validation = cross_validate(linear_regressor, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

# Predicciones del modelo
linear_regression_predictions = linear_regressor.predict(X_test)
linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
print(f'Mean Squared Error - Linear Regression: {linear_regression_mse}')

# 2. **Red Neuronal (MLP)**
mlp = MLPRegressor(hidden_layer_sizes=(5,5), activation='relu')
mlp.fit(X_train, y_train)
mlp_validation = cross_validate(mlp, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

# Predicciones del modelo
mlp_predictions = mlp.predict(X_test)
mlp_mse = mean_squared_error(y_test, mlp_predictions)
print(f'Mean Squared Error - MLP Regressor: {mlp_mse}')

# 3. **Gradient Boosting Regressor**
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_validation = cross_validate(gb, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

# Predicciones del modelo
gb_predictions = gb.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_predictions)
print(f'Mean Squared Error - Gradient Boosting: {gb_mse}')

# Mostrar resultados en un DataFrame
print("Imprimiendo resultados de los modelos")
results = {
    'Linear Regression': [linear_regression_mse],
    'ReLU Neural Network': [mlp_mse],
    'Gradient Boosting Regressor': [gb_mse]
}

modeling_results = pd.DataFrame(data=results, index=['MSE'])
print(modeling_results)


# Crear subplots para graficar los residuales
fig1, (LR, FNN, GBR) = plt.subplots(1, 3, figsize=(15, 5))
fig1.tight_layout()

# Graficar residuales para cada modelo
LR.scatter(x=linear_regression_predictions, y=y_test - linear_regression_predictions, color='red', alpha=0.06)
FNN.scatter(x=mlp_predictions, y=y_test - mlp_predictions, color='green', alpha=0.06)
GBR.scatter(x=gb_predictions, y=y_test - gb_predictions, color='blue', alpha=0.06)

# Etiquetas de los ejes
LR.set_xlabel('Linear Regression Predicted Fraud Score')
FNN.set_xlabel('Neural Network Predicted Fraud Score')
GBR.set_xlabel('Gradient Boosting Predicted Fraud Score')

LR.set_ylabel('Linear Regression Residual')
FNN.set_ylabel('Neural Network Residual')
GBR.set_ylabel('Gradient Boosting Residual')

# Mostrar gráficos
plt.show()
