import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats.mstats import winsorize
import os
import sys

# CARICAMENTO DATASET
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
file_path = os.path.join(script_dir, "normalizzazione_dataset.py")
print(os.getcwd())

df = pd.read_csv(r"C:\Users\FrancescoFenzi\repo_git\EconML-Classifier\Algoritmi\online_news_PRE_normalizzazione.csv")

X = df.drop(['classe'], axis=1)
y = df['classe']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)

# Valutazione del modello iniziale
clf.fit(X_train, y_train)

y_pred_initial = clf.predict(X_test)
print(f'Initial Accuracy: {accuracy_score(y_test, y_pred_initial)}')
print(classification_report(y_test, y_pred_initial))

""" param_grid = {
    'n_estimators': [10, 50, 100], # numero di alberi nella foresta
    'max_depth': [None, 10, 20, 30], # max_depth: profondità massima dell'albero
    'min_samples_split': [2, 5, 10], # min_samples_split: numero minimo di campioni richiesti per suddividere un nodo
    'min_samples_leaf': [1, 2, 4], # min_sample_leaf : numero minimo di campioni che un nodo foglia deve contenere
    'max_features': ['auto', 'sqrt', 'log2'] # max_features : numero massimo di caratteristiche da considerare per trovare la migliore divisione
}

# Inizializzazione del GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train) """

""" GridSearchCV : È uno strumento di scikit-learn che permette di effettuare una ricerca attraverso una griglia specificata di parametri per un modello.

    estimator : Il modello che si vuole ottimizzare, in questo caso un SVC (Support Vector Classifier).
    param_grid : La griglia di iperparametri definita in precedenza.
    
    cv : Cross-validation, il numero di suddivisioni del dataset da usare per la validazione incrociata. 
    Qui, cv=5 indica che si utilizzerà una cross-validation a 5 fold.
    scoring: La metrica di valutazione da ottimizzare. Qui viene utilizzata l'accuratezza (accuracy).
   
    n_jobs: Il numero di job da eseguire in parallelo. -1 significa che verranno utilizzati tutti i processori disponibili.
"""

# Predizione sui dati di test con il miglior modello
""" best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Valutazione del modello
print(f'Best parameters: {grid_search.best_params_}')
print(f'Optimized Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# OOB (Out-of-Bag) error rate
best_clf.set_params(oob_score=True)
best_clf.fit(X_train, y_train)

min_estimators = 15
max_estimators = 1000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    best_clf.set_params(n_estimators=i)
    best_clf.fit(X_train, y_train)
    oob_error = 1 - best_clf.oob_score_
    error_rate[i] = oob_error

oob_series = pd.Series(error_rate)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('#fafafa')
oob_series.plot(kind='line', color='red')
plt.axhline(0.04, color='#875FDB', linestyle='--')
plt.axhline(0.05, color='#875FDB', linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')
plt.show()

print('OOB Error rate for 400 trees is: {0:.5f}'.format(oob_series[400]))

# imposteremo il numero di alberi calcolato utilizzando
# il tasso di errore OOB, rimuovendo i parametri warm_start e oob_score. 
# Includeremo inoltre il parametro bootstrap
clf.set_params(n_estimators=400,
                  bootstrap = True,
                  warm_start=False,
                  oob_score=False)

# Predizione finale sui dati di test
y_pred_final = best_clf.predict(X_test)

# Valutazione del modello finale
print(f'Final Accuracy: {accuracy_score(y_test, y_pred_final)}')
print(classification_report(y_test, y_pred_final)) """