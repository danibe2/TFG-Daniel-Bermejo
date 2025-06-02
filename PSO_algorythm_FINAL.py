import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# 1. Cargar el dataset
data = pd.read_csv('base.csv')

# 2. Ver distribución original
conteo = data['fraud_bool'].value_counts()
porcentaje = data['fraud_bool'].value_counts(normalize=True) * 100

# 3. Balancear el dataset (50/50 con submuestreo)
fraud = data[data['fraud_bool'] == 1]
non_fraud = data[data['fraud_bool'] == 0].sample(n=len(fraud), random_state=42)
balanced_df = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Separar variables predictoras (X) y objetivo (y)
X = balanced_df.drop(columns=['fraud_bool'])
y = balanced_df['fraud_bool']

# 5. Codificar variables categóricas con One-Hot Encoding
categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
X = pd.get_dummies(X, columns=categorical_cols)

# 6. Escalar todas las columnas numéricas
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 7. División en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 3. Definición del PSO (Modificado para incluir alpha)
activation_functions = ['relu', 'tanh', 'logistic']

class Particle:
    def __init__(self):
        # Posición ahora incluye layer1, layer2, alpha, activation
        self.position = np.array([
            np.random.uniform(16, 128), # layer1
            np.random.uniform(0, 64),   # layer2
            np.random.uniform(1e-5, 1.0), # alpha (ejemplo de rango)
            np.random.uniform(0, len(activation_functions) - 1)  # activation index
        ])
        self.velocity = np.random.uniform(-1, 1, 4) # 4 dimensiones
        self.best_position = self.position.copy()
        self.best_score = 0

    def evaluate(self):
        layer1 = max(1, int(round(self.position[0])))
        layer2 = int(round(self.position[1]))
        alpha = self.position[2] # Nuevo: valor de alpha
        activation_index = int(round(self.position[3]))
        activation = activation_functions[activation_index % len(activation_functions)]  # prevención de overflow

        hidden_layers = (layer1,) if layer2 == 0 else (layer1, layer2)

        # Incluir alpha en el modelo
        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation,
                              alpha=alpha, max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = f1_score(y_test, preds)

        if score > self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()

        print(f"Evaluando {hidden_layers}, alpha={alpha:.5f}, activation={activation} -> F1 Score: {score:.4f}")
        return score

    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        self.position += self.velocity
        # Ajustar el clipping para 3 dimensiones
        self.position = np.clip(self.position, [16, 0, 1e-5,0], [128, 64, 1.0,len(activation_functions) - 1])

# 4. Ejecutar PSO (Ajustar global_best_position a 3 dimensiones)
NUM_PARTICLES = 20
NUM_ITERATIONS = 20

swarm = [Particle() for _ in range(NUM_PARTICLES)]
global_best_position = swarm[0].position.copy() # Ahora 3 dimensiones
global_best_score = 0

for iteration in range(NUM_ITERATIONS):
    print(f"\nIteración {iteration + 1}")
    for particle in swarm:
        score = particle.evaluate()
        if score > global_best_score:
            global_best_score = score
            global_best_position = particle.position.copy()

    for particle in swarm:
        particle.update_velocity(global_best_position)
        particle.update_position()

# Resultado final (Ajustar para mostrar alpha)
layer1 = int(round(global_best_position[0]))
layer2 = int(round(global_best_position[1]))
alpha = global_best_position[2]
activation = activation_functions[int(round(global_best_position[3])) % len(activation_functions)]
best_layers = (layer1,) if layer2 == 0 else (layer1, layer2)
print(f"\nMejor arquitectura encontrada por PSO: {best_layers}, alpha={alpha:.5f}, activation={activation} con F1 Score = {global_best_score:.4f}")