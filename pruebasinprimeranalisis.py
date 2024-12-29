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
from sklearn.preprocessing import StandardScaler          #importing scikit-learn's standard scaler

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

boxscores = boxscores.sample(frac=0.1, random_state=42)

# Selección de variables (features) y target
X = boxscores[['velocity_6h', 'customer_age', 'bank_months_count', 'proposed_credit_limit', 'credit_risk_score', 'name_email_similarity', 'zip_count_4w']]  # características
y = boxscores['fraud_score']  # objetivo

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. **Regresión Lineal**
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_scaled, y_train)
linear_regression_validation = cross_validate(linear_regressor, X_train_scaled, y_train, cv=3, return_train_score=True, return_estimator=True)

# 2. **Red Neuronal (MLP)**
mlp = MLPRegressor(hidden_layer_sizes=(3, 3), max_iter=1000, random_state=42, learning_rate_init=0.01)  # Ajustes para mejorar convergencia
mlp.fit(X_train_scaled, y_train)
mlp_validation = cross_validate(mlp, X_train_scaled, y_train, cv=3, return_train_score=True, return_estimator=True)

# 3. **Gradient Boosting Regressor**
gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)  
gb.fit(X_train_scaled, y_train)
gb_validation = cross_validate(gb, X_train_scaled, y_train, cv=3, return_train_score=True, return_estimator=True)

linear_regression_predictions = linear_regressor.predict(X_test_scaled)                              #make predictions based on the test set for the linear regression model
mlp_predictions = mlp.predict(X_test_scaled)                                                         #make predictions based on the test set for the neural network model
gb_predictions = gb.predict(X_test_scaled)                                                           #make predictions based on the test set for the gradient boosting model

linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)             #calculate the MSE for the linear regression model
mlp_mse = mean_squared_error(y_test, mlp_predictions)                                         #calculate the MSE for the neural network model
gb_mse = mean_squared_error(y_test, gb_predictions)                                           #calculate the MSE for the gradient boosting model

print("Linear Regression MSE:", linear_regression_mse)
print("Neural Network MSE:", mlp_mse)
print("Gradient Boosting MSE:", gb_mse)

results = {'Linear Regression':[linear_regression_mse],'ReLU Neural Network':[mlp_mse],'Gradient Boosting Regressor':[gb_mse]}
modeling_results = pd.DataFrame(data=results, index=['MSE'])

print(modeling_results)

# Crear subplots para graficar los residuales
fig, (LR, FNN, GBR) = plt.subplots(1, 3, figsize=(15, 5))
fig.tight_layout()

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