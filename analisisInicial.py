import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset
df = pd.read_csv("base.csv")

# Calcular distribución porcentual
fraud_dist = df['fraud_bool'].value_counts(normalize=True) * 100

# Mostrar resultados
print(f"No Fraude (0): {fraud_dist[0]:.2f}%")
print(f"Fraude (1): {fraud_dist[1]:.2f}%")

# 2. Tipos de datos y resumen estadístico
print("\n Tipos de datos:")
print(df.dtypes)

print("\n Resumen estadístico general:")
print(df.describe(include='all'))

# 3. Histogramas de variables numéricas
print("\n Histogramas:")
df.select_dtypes(include=['int64', 'float64']).hist(figsize=(16, 12), bins=30)
plt.tight_layout()
plt.show()

# 4. Boxplots para detectar outliers
print("\n Boxplots para variables numéricas:")
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.show()

# 5. Matriz de correlación
print("\n Matriz de correlación:")
plt.figure(figsize=(14, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

# 6. Distribución de la variable objetivo
print("\n Distribución de la variable 'fraud_bool':")
print(df['fraud_bool'].value_counts(normalize=True) * 100)

# 7. KDEs por clase para cada variable numérica
print("\n Comparación de variables por clase 'fraud_bool':")
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    if col != 'fraud_bool':
        plt.figure(figsize=(6, 3))
        sns.kdeplot(data=df, x=col, hue='fraud_bool', common_norm=False)
        plt.title(f'Distribución de {col} por clase")
        plt.tight_layout()
        plt.show()
