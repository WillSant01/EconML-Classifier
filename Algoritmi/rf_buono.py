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

df = pd.read_csv(r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\OnlineNewsPopularity.csv')
df.columns = df.columns.str.lstrip() # tolgo gli spazi negli indici
df_brutta = df.copy()


# A causa delle 61 colonne, bisogna disabilitare il limite del display.
pd.set_option('display.max_columns', None)

# Decidiamo di togliere:
# - La colonna degli URL (gli indici identificano l'articolo)
# - La colonna IS_WEEKEND (abbiamo già le colonne per i giorni della settimana -> ridondanza)
# - La colonna TIMEDELTA (inutile per il business goal)
# - La colonne ABS (ridondanza di dati)

nomi = df_brutta.columns.to_list()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# RIMOZIONE DELLE COLONNE NON UTILI

df_brutta.drop(['url', 'timedelta', 'is_weekend', 'title_subjectivity',
               'title_sentiment_polarity'], axis=1, inplace=True)

df_brutta["kw_min"] = df_brutta[["kw_min_min", "kw_min_max", "kw_min_avg"]].median(axis=1)
df_brutta["kw_max"] = df_brutta[["kw_max_min", "kw_max_max", "kw_max_avg"]].median(axis=1)
df_brutta["kw_avg"] = df_brutta[["kw_avg_min", "kw_avg_max", "kw_avg_avg"]].median(axis=1)

df_brutta.drop(["kw_min_min", "kw_min_max", "kw_min_avg", "kw_max_min", "kw_max_max", "kw_max_avg"], axis=1, inplace=True)

nomi = df_brutta.columns.to_list()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# FEATURE CONSTRUCTION

# Vogliamo gestire le colonne di true/false accorpandole in un'unica con un range di valori (1 - n_colonne_binarie)
def gestione_binario(data, keyword):
    # Prendo i nomi delle colonne binarie che mi interessano
    colonne_da_accorpare = [
        col for col in data.columns if data[col].nunique() == 2 and keyword in col]
    # Estraggo dal dataframe
    matrice_da_accorpare = data[colonne_da_accorpare].copy()

    # Creo una series con la mappatura dei valori in base alla posizione
    colonna_nuova = pd.Series(matrice_da_accorpare.values.argmax(axis=1) + 1)

    # Li tolgo dal df e aggiungo la colonna
    data.drop(columns=colonne_da_accorpare, inplace=True)
    data[colonna_nuova.name] = colonna_nuova

    data.rename(columns={None: keyword}, inplace=True)

    return data


# Per i topic
df_brutta = gestione_binario(df_brutta, "data_channel")
# Per i giorni della settimana
df_brutta = gestione_binario(df_brutta, "weekday")

nomi = df_brutta.columns.to_list()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# OTTIMIZZAZIONE DELLA MEMORIA

# Ciclo sulle colonne e stampo i conteggi.
for col in nomi:
    print(f"{col} : {df_brutta[col].nunique()}")
    

df_brutta.dtypes

df_brutta.memory_usage(deep=True)/1048576
(df_brutta.memory_usage(deep=True)/1048576).sum()


def controllo_int(series):
    for num in series:
        if num != int(num):
            return True
    return False

check = df_brutta.apply(controllo_int)

box = []

for col_name, non_decimal in zip(nomi, check):
    if not non_decimal:
        min_val = df_brutta[col_name].min()
        max_val = df_brutta[col_name].max()
        box.append([col_name, min_val, max_val])

tipo_check = pd.DataFrame(box, columns=['Nome', 'Minimo', 'Massimo'])
tipo_check.set_index('Nome', inplace=True)
        
for index, row in tipo_check.iterrows():
    if -128 <= row[0] and row[1] <= 127:
        df_brutta[index] = df_brutta[index].astype('int8')
    elif -32768 <= row[0] and row[1] <= 32767:
        df_brutta[index] = df_brutta[index].astype('int16')
    elif -2147483648 <= row[0] and row[1] <= 2147483647:
        df_brutta[index] = df_brutta[index].astype('int32')
    else:
        df_brutta[index] = df_brutta[index].astype('int64')

df_brutta.dtypes
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MAPPATURA DEL LABEL

X_winsorized = df_brutta.copy()

for column in X_winsorized.columns:
    X_winsorized[column] = winsorize(df_brutta[column], limits=[0.05, 0.05])

X_winsorized['shares'].describe()
df_brutta['shares'].describe()

plt.figure(figsize=(12, 6))
sns.histplot(X_winsorized['shares'], kde=True)

"""
def classificazione_shares(shares):
    T1 = np.percentile(shares, 30)
    T2 = np.percentile(shares, 60)
    
    if shares <= T1:
        return 1
    elif shares < T2:
        return 2
    else:
        return 3
"""
def classificazione_shares(shares):
    
    if shares <= 900:
        return 1
    elif shares < 4250:
        return 2
    else:
        return 3

# Applica la funzione di classificazione al DataFrame

df_brutta['classe'] = df_brutta['shares'].apply(classificazione_shares)
X_winsorized['classe'] = X_winsorized['shares'].apply(classificazione_shares)

df_brutta.drop(columns="shares", inplace=True)
X_winsorized.drop(columns="shares", inplace=True)

nomi = df_brutta.columns.to_list()

X = X_winsorized.drop('classe', axis=1)
y = X_winsorized['classe']


# Scaling dei dati winsorizzati
scaler_w = RobustScaler()
scaled_data_winsorized = scaler_w.fit_transform(X_winsorized)

pca = PCA()
X_pca = pca.fit_transform(scaled_data_winsorized)

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

pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)

# Valutazione del modello iniziale
clf.fit(X_train, y_train)

y_pred_initial = clf.predict(X_test)
print(f'Initial Accuracy: {accuracy_score(y_test, y_pred_initial)}')
print(classification_report(y_test, y_pred_initial))

param_grid = {
    'n_estimators': [10, 50, 100], # numero di alberi nella foresta
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