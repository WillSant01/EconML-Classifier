import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from urllib.request import urlopen
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carichiamo il dataset
data = pd.read_csv(r"C:\Users\FrancescoFenzi\repo_git\EconML-Classifier\Algoritmi\online_news_RF.csv")

plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='shares', kde=True)
plt.show()

# Mappatura della label
def classificazione_shares(shares):
    if shares < 900:
        return 1
    elif shares < 1500:
        return 2
    elif shares < 3000:
        return 3
    else:
        return 4


data['classe'] = data['shares'].apply(classificazione_shares)
data.drop(['shares'], axis=1)

print(data['classe'].value_counts(normalize=True) * 100)

# Separazione delle feature e dell'etichetta
X = data.drop('classe', axis=1)  # tutte le colonne eccetto 'successo'
y = data['classe']  # la colonna 'successo'

pca = PCA()
X_pca = pca.fit_transform(X)

explained_variance_ratio = pca.explained_variance_ratio_

# Calcola la varianza cumulativa spiegata
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Traccia lo Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1),
         cumulative_variance_ratio, marker='o', linestyle='-', color='b')
plt.xlabel('Numero di componenti principali')
plt.ylabel('Varianza cumulativa spiegata')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)

""" smote = SMOTE()
X_sampled, y_sampled = smote.fit_resample(X_pca, y) """ 

# Divisione del dataset in training e testing set
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Inizializzazione del modello RandomForestClassifier
clf = RandomForestClassifier(random_state=42)

# Valutazione del modello iniziale
clf.fit(X_train, y_train)
y_pred_initial = clf.predict(X_test)
print(f'Initial Accuracy: {accuracy_score(y_test, y_pred_initial)}')
print(classification_report(y_test, y_pred_initial))

# 91% accurasy con smote e n_component = 8
# 92% accuracy senza smote e n_component = 6

# Tuning (ottimizzazione)

# Definizione della griglia di iperparametri
param_grid = {
    'n_estimators': [10,50,100], # numero di alberi nella foresta
    'max_depth': [None, 10, 20, 30], # max_depth: profondità massima dell'albero
    'min_samples_split': [2, 5, 10], # min_samples_split: numero minimo di campioni richiesti per suddividere un nodo
    'min_samples_leaf': [1, 2, 4], # min_sample_leaf : numero minimo di campioni che un nodo foglia deve contenere
    'max_features': ['auto', 'sqrt', 'log2'] # max_features : numero massimo di caratteristiche da considerare per trovare la migliore divisione
}

# Inizializzazione del GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

""" GridSearchCV : È uno strumento di scikit-learn che permette di effettuare una ricerca attraverso una griglia specificata di parametri per un modello.

    estimator : Il modello che si vuole ottimizzare, in questo caso un SVC (Support Vector Classifier).
    param_grid : La griglia di iperparametri definita in precedenza.
    
    cv : Cross-validation, il numero di suddivisioni del dataset da usare per la validazione incrociata. 
    Qui, cv=5 indica che si utilizzerà una cross-validation a 5 fold.
    scoring: La metrica di valutazione da ottimizzare. Qui viene utilizzata l'accuratezza (accuracy).
   
    n_jobs: Il numero di job da eseguire in parallelo. -1 significa che verranno utilizzati tutti i processori disponibili.
"""

# Predizione sui dati di test con il miglior modello
best_clf = grid_search.best_estimator_
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
print(classification_report(y_test, y_pred_final))