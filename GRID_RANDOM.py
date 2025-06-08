import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from scipy.stats import uniform, randint

# ======================
# 1. Cargar y preparar el dataset
# ======================
data = pd.read_csv('base.csv')  # Asegúrate que el nombre del archivo esté correcto

# Balancear las clases (submuestreo 50/50)
fraud = data[data['fraud_bool'] == 1]
non_fraud = data[data['fraud_bool'] == 0].sample(n=len(fraud), random_state=42)
balanced_df = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

# Separar X e y
X = balanced_df.drop(columns=['fraud_bool'])
y = balanced_df['fraud_bool']

# Codificar variables categóricas
categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
X = pd.get_dummies(X, columns=categorical_cols)

# Escalar variables numéricas
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# 2. Grid Search para MLPClassifier
# ======================
mlp = MLPClassifier(max_iter=500, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

grid_search = GridSearchCV(mlp, param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
y_pred_grid = grid_search.predict(X_test)

print("\n=== Grid Search ===")
print("Best Params:", grid_search.best_params_)
print("F1 Score:", f1_score(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))

# ======================
# 3. Random Search para MLPClassifier
# ======================
param_dist = {
    'hidden_layer_sizes': [(randint.rvs(30, 150),), (randint.rvs(30, 150), randint.rvs(30, 150))],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate': ['constant', 'adaptive']
}

random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=30,
                                   scoring='f1', cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
y_pred_rand = random_search.predict(X_test)

print("\n=== Random Search ===")
print("Best Params:", random_search.best_params_)
print("F1 Score:", f1_score(y_test, y_pred_rand))
print(classification_report(y_test, y_pred_rand))
