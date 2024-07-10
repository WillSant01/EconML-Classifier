import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA


"""
FEATURE ENGINEERING

FEATURE TRANSFORMATION:
    1) Missing value imputation: utilizzare una funzione matematica per sostituire gli na.
    2) Outlier detection: trovarli e eliminarli
    3) Feature scaling: Impostare una scala di valori per determinate colonne.

FEATURE CONSTRUCTION: aggrego feature creando una nuova.

FEATURE SELECTION: scarto le feature inutili per il label.

FEATURE EXTRACTION: (PCA) crea nuove feature in base alla varianza.

"""

"""
FEATURE SCALING: STANDARDIZATION
    Accomuna le scale di valori delle feature indipendenti.
    
    Z - Score: X_i' = (X_i - media_X) / std
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler() --> scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = sclaer.transform(X_test)


FEATURE SCALING: NORMALIZATION
    Una scala comune senza perdere informazioni.
    
    1) MinMaxScaling: X_i' = (X_i - min_x) / (max_X - min_X) --> [0 : 1]: 0 minimo, 1 massimo
    2) Mean Normalization: X_i' = (X_i - media_X) / (max_X - min_X) --> [-1 : 1]: media in centro
    3) Max Abs Scaling: X_i' = X_i / |max_X|
    4) Robust Scaling: X_i' = (X_i - media_X) / IQR --> 

"""

"""
MATHEMATICAL TRASNFORMATIONS
    sns.distplot, q-q plot, pd.skew() = 0
    1) Function trasform:
        1.1) Logaritmica
        1.2) Reciproca
        1.3) Quadrato
    2) Trasfromazione reciproca
    3) Power trasformation.

"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# CARICAMENTO DATASET

df = pd.read_csv(r'C:\Users\WilliamSanteramo\repo_github\EconML-Classifier\OnlineNewsPopularity.csv')
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

df_brutta.drop(['url', 'timedelta', 'is_weekend'], axis=1, inplace=True)

df_brutta["kw_min"] = df_brutta[["kw_min_min", "kw_min_max", "kw_min_avg"]].median(axis=1)
df_brutta["kw_max"] = df_brutta[["kw_max_min", "kw_max_max", "kw_max_avg"]].median(axis=1)
df_brutta["kw_avg"] = df_brutta[["kw_avg_min", "kw_avg_max", "kw_avg_avg"]].median(axis=1)

df_brutta.drop(["kw_min_min", "kw_min_max", "kw_min_avg", "kw_max_min", "kw_max_max",
               "kw_max_avg", "kw_avg_min", "kw_avg_max", "kw_avg_avg"], axis=1, inplace=True)

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

df_cart = df_brutta.copy()
df_rf = df_brutta.copy()


def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    column_cleaned = column[(column >= lower_bound) & (column <= upper_bound)]
    return column_cleaned

df_cart = remove_outliers(df_cart)

df_cart = df_cart.dropna()

for column in df_rf.columns:
    df_rf[column] = winsorize(df_rf[column], limits=[0.05, 0.05])

df_cart.to_csv(r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\Algoritmi\online_news_CART', index=False)

df_rf.to_csv(r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\Algoritmi\online_news_RF', index=False)