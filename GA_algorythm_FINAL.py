import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import random

# 1. Cargar el dataset
data = pd.read_csv('base.csv')

# 2. Balancear el dataset (50/50 con submuestreo)
fraud = data[data['fraud_bool'] == 1]
non_fraud = data[data['fraud_bool'] == 0].sample(n=len(fraud), random_state=42)
balanced_df = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Separar variables predictoras y etiquetas
X = balanced_df.drop(columns=['fraud_bool'])
y = balanced_df['fraud_bool']

# 4. Codificar variables categóricas
categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
X = pd.get_dummies(X, columns=categorical_cols)

# 5. Escalar numéricas
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 6. División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 7. GA para arquitectura + alpha
activation_functions = ['relu', 'tanh', 'logistic']

def create_individual():
    return [
        random.randint(16, 128),     # capa 1
        random.randint(0, 64),     # capa 2 (0 = no usar)
        random.choice(activation_functions),  # activación
        10 ** random.uniform(-5, 0)  # alpha en rango [1e-5, 1]
    ]

def mutate(ind):
    if random.random() < MUTATION_RATE:
        ind[0] = random.randint(16, 128)
    if random.random() < MUTATION_RATE:
        ind[1] = random.randint(0, 64)
    if random.random() < MUTATION_RATE - 0.2:
        ind[2] = random.choice(activation_functions)
    if random.random() < MUTATION_RATE - 0.1:
        ind[3] = 10 ** random.uniform(-5, 0)
    return ind

def crossover(p1, p2):
    return [
        p1[0] if random.random() < CROSS_RATE else p2[0],
        p1[1] if random.random() < CROSS_RATE else p2[1],
        p1[2] if random.random() < CROSS_RATE else p2[2],
        p1[3] if random.random() < CROSS_RATE else p2[3]
    ]

def evaluate(ind):
    layers = (ind[0],) if ind[1] == 0 else (ind[0], ind[1])
    activation = ind[2]
    alpha = ind[3]

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation=activation,
        alpha=alpha,
        max_iter=2000,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"Evaluando {ind} -> F1 Score: {f1:.4f}")
    return f1

# 8. Ejecutar GA
GENERATIONS = 20
POP_SIZE = 20
CROSS_RATE = 0.5
MUTATION_RATE = 0.5

population = [create_individual() for _ in range(POP_SIZE)]

for generation in range(GENERATIONS):
    print(f"\nGeneración {generation + 1}")
    scores = [(evaluate(ind), ind) for ind in population]
    scores.sort(reverse=True)
    best_score = scores[0][0]
    print(f"Mejor F1 Score: {best_score:.4f}")

    survivors = [ind for _, ind in scores[:POP_SIZE // 2]]
    offspring = []
    while len(offspring) < POP_SIZE // 2:
        p1, p2 = random.sample(survivors, 2)
        child = crossover(p1, p2)
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        offspring.append(child)

    population = survivors + offspring

# 9. Mostrar mejor configuración
best_individual = scores[0][1]
best_layers = (best_individual[0],) if best_individual[1] == 0 else (best_individual[0], best_individual[1])
print(f"\nMejor individuo encontrado: capas={best_layers}, activación={best_individual[2]}, alpha={best_individual[3]:.5f}")
