import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from scipy.stats import loguniform

data = pd.read_csv('base.csv')

# Balancear clases 50/50
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
# Función para generar combinaciones para Grid Search
# ======================
def generate_grid_layer_configs(layer1_vals=[32, 64, 96, 128], layer2_vals=[0, 16, 32, 48, 64]):
    configs = []
    for l1 in layer1_vals:
        for l2 in layer2_vals:
            if l2 <= 1:
                configs.append((l1,))
            else:
                configs.append((l1, l2))
    return configs

# ======================
# Paso 2: Grid Search
# ======================
activation_functions = ['relu', 'tanh', 'logistic']
alphas = [1e-5, 1e-3, 1e-1]
layer_configs = generate_grid_layer_configs()

param_grid = {
    'hidden_layer_sizes': layer_configs,
    'alpha': alphas,
    'activation': activation_functions
}

grid_search = GridSearchCV(
    estimator=MLPClassifier(max_iter=500, random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

y_pred_grid = grid_search.predict(X_test)
print("\n=== Resultados Grid Search ===")
print("Mejores parámetros:", grid_search.best_params_)
print("F1 Score:", f1_score(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))

# ======================
# Paso 3: Random Search
# ======================
print("\n=== Paso 3: Iniciando Random Search ===")
def generate_random_layers(n_configs=30):
    configs = []
    for _ in range(n_configs):
        layer1 = int(np.random.uniform(16, 128))
        layer2 = int(np.random.uniform(0, 64))
        if layer2 <= 1:
            configs.append((layer1,))
        else:
            configs.append((layer1, layer2))
    return configs

param_dist = {
    'hidden_layer_sizes': generate_random_layers(30),
    'alpha': loguniform(1e-5, 1.0),
    'activation': activation_functions
}

print(f"Probando 30 configuraciones aleatorias de arquitectura ocultas...")
random_search = RandomizedSearchCV(
    estimator=MLPClassifier(max_iter=500, random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1
)

print("Entrenando modelos con Random Search (esto también puede tardar)...")
random_search.fit(X_train, y_train)
print("Random Search completado.")

y_pred_rand = random_search.predict(X_test)
print("\n=== Resultados Random Search ===")
print("Mejores parámetros:", random_search.best_params_)
print("F1 Score:", f1_score(y_test, y_pred_rand))
print(classification_report(y_test, y_pred_rand))
