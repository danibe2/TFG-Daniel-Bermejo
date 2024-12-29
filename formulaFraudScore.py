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
