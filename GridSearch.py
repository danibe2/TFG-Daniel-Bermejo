import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import itertools

# === 1. Cargar y preprocesar ===
data = pd.read_csv('base.csv')
fraud = data[data['fraud_bool'] == 1]
non_fraud = data[data['fraud_bool'] == 0].sample(n=len(fraud), random_state=42)
df = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=['fraud_bool'])
y = df['fraud_bool']
X = pd.get_dummies(X, columns=['payment_type', 'employment_status', 'housing_status', 'source', 'device_os'])
X[X.select_dtypes(include=['int64', 'float64']).columns] = StandardScaler().fit_transform(
    X.select_dtypes(include=['int64', 'float64']))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === 2. espacio de búsqueda ===
param_grid = {
    'layer1': list(range(16, 129, 16)),  
    'layer2': list(range(0, 65, 16)),    
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [10 ** x for x in np.linspace(-5, 0, 10)] 
}

# === 3. Grid Search ===
combinaciones = list(itertools.product(*param_grid.values()))
best_f1 = -1
best_params = None

for l1, l2, activation, alpha in combinaciones:
    layers = (l1,) if l2 == 0 else (l1, l2)
    model = MLPClassifier(hidden_layer_sizes=layers, activation=activation,
                          alpha=alpha, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred)
    
    print(f"capas={layers}, act={activation}, alpha={alpha:.5f} -> F1={f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_params = (layers, activation, alpha)

# === 4. Resultados ===
print("\n=== MEJOR CONFIGURACIÓN GRID SEARCH ===")
print(f"Capas: {best_params[0]}")
print(f"Activación: {best_params[1]}")
print(f"Alpha: {best_params[2]:.5f}")
print(f"F1 Score: {best_f1:.4f}")

