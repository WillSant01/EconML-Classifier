import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
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

df_winsorized = df_brutta.copy()

for column in df_winsorized.columns:
    df_winsorized[column] = winsorize(df_brutta[column], limits=[0.05, 0.05])

df_winsorized['shares'].describe()
df_brutta['shares'].describe()

plt.figure(figsize=(12, 6))
sns.histplot(df_winsorized['shares'], kde=True)

def classificazione_shares(shares):
    if shares < 1400:
        return 1
    elif shares < 4250:
        return 2
    else:
        return 3

# Applica la funzione di classificazione al DataFrame

df_brutta['classe'] = df_brutta['shares'].apply(classificazione_shares)
df_winsorized['classe'] = df_winsorized['shares'].apply(classificazione_shares)

df_brutta.drop(columns="shares", inplace=True)
df_winsorized.drop(columns="shares", inplace=True)

nomi = df_brutta.columns.to_list()

X = df_winsorized.drop('classe', axis=1)
y = df_winsorized['classe']


# Scaling dei dati winsorizzati
scaler_w = RobustScaler()
X_scaled = scaler_w.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

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

# Dividi in set di addestramento e test
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_pca, y, test_size=0.2, random_state=42)

clf_w = DecisionTreeClassifier(criterion='gini', max_depth=4)
clf_w.fit(X_train_w, y_train_w)


y_pred_w = clf_w.predict(X_test_w)
accuracy_w = accuracy_score(y_test_w, y_pred_w)
print("Accuracy con winsorization pre mappatura:", accuracy_w)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

