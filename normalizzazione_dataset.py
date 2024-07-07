import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

"""
DOMANDE BASE:
    1) Quanto è grande il dataset? df.shape()
    2) Cosa assomiglia? df.head()
    3) Tipologie delle colonne? df.info, modifica la tipologia per ottimizzare la memoria
    4) Ci sono valori nulli? df.isnull().sum()
    5) Matematicamente che aspetto hanno? df.describe()
    6) Ci sono valori duplicati? df.duplicated().sum()
    7) Ci sono correlazioni tra le colonne? df.corr()

ANALISI UNI-VARIATA PER DATI NUMERICI:
    1) Istogramma. plt.hist(df["colonna"], bins = num)
    2) Distplot. sns.distplot(df["colonna"])
    3) Boxplot. sns.boxplot(df["colonna"])
    
ANALISI BI-VARIATA TRA NUMERICO E NUMERICO:
    1) Scatter-plot. sns.scatterplot(df["colonna1"], df["colonna2"], hue = df["colonna3"])

PANDAS PROFILING :
    pip install pandas-profiling
    from pandas_profiling import ProfileReport
    
    report = ProfileReport(df)
    prof.to_file(output_file = "EDA_online_news.html")
"""

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

df = pd.read_csv(r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\OnlineNewsPopularity.csv')
df.columns = df.columns.str.lstrip() # tolgo gli spazi negli indici
df_brutta = df.copy()


# A causa delle 61 colonne, bisogna disabilitare il limite del display.
pd.set_option('display.max_columns', None)

print(df_brutta.head())
# Decidiamo di togliere:
# - La colonna degli URL (gli indici identificano l'articolo)
# - La colonna IS_WEEKEND (abbiamo già le colonne per i giorni della settimana -> ridondanza)
# - La colonna TIMEDELTA (inutile per il business goal)
# - La colonne ABS (ridondanza di dati)

nomi = df_brutta.columns.to_list()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# RIMOZIONE DELLE COLONNE NON UTILI

df_brutta.drop(['url', 'timedelta', 'is_weekend', 'abs_title_subjectivity',
               'abs_title_sentiment_polarity'], axis=1, inplace=True)

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MAPPATURA DEL LABEL

soglia_bassa = 900

soglia_alta = 4250

# Definisci le classi

def classificazione_shares(shares):
    if shares < soglia_bassa:
        return 1
    elif shares < soglia_alta:
        return 2
    else:
        return 3

# Applica la funzione di classificazione al DataFrame

df_brutta['classe'] = df_brutta['shares'].apply(classificazione_shares)

df_brutta.drop(columns="shares", inplace=True)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# OTTIMIZZAZIONE DELLA MEMORIA

# Ciclo sulle colonne e stampo i conteggi.
for col in nomi:
    print(f"{col} : {df_brutta[col].nunique()}")
    

df_brutta.dtypes

df_brutta.memory_usage(deep=True)/1048576
(df_brutta.memory_usage(deep=True)/1048576).sum()

nomi = df_brutta.columns.to_list()

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

Tipo_check = pd.DataFrame(box, columns=['Nome', 'Minimo', 'Massimo'])
Tipo_check.set_index('Nome', inplace=True)
        
for index, row in Tipo_check.iterrows():
    if -128 <= row[0] and row[1] <= 127:
        df_brutta[index] = df_brutta[index].astype('int8')
    elif -32768 <= row[0] and row[1] <= 32767:
        df_brutta[index] = df_brutta[index].astype('int16')
    elif -2147483648 <= row[0] and row[1] <= 2147483647:
        df_brutta[index] = df_brutta[index].astype('int32')
    else:
        df_brutta[index] = df_brutta[index].astype('int64')

    
df_brutta['n_tokens_title'] = df_brutta['n_tokens_title'].astype('int8')
df_brutta['classe'] = df_brutta['classe'].astype('int8')
df_brutta['data_channel'] = df_brutta['data_channel'].astype('int8')
df_brutta['weekday'] = df_brutta['weekday'].astype('int8')





df_brutta.to_csv(r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\Algoritmi\online_news_outliers.csv', index=False)
















"""

def count_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers)


outliers_count = df_brutta.apply(count_outliers)


# print(outliers_count)

# Calcola la percentuale di outlier per ogni colonna
outliers_percentage = (outliers_count / len(df_brutta)) * 100

# Aggiungo il simbolo '%'
outliers_percentage = outliers_percentage.map(lambda x: f'{x:.2f}%')

# Visualizzazione Risultati

outliers_summary = pd.DataFrame({
    'Numero di Outliers': outliers_count,
    'Percentuale di Outliers': outliers_percentage})

# Stampa il risultato
print(outliers_summary)


def rimuovi_outliers(df, soglia_percentuale):
    q_low = df.quantile(soglia_percentuale / 100)
    q_high = df.quantile(1 - (soglia_percentuale / 100))
    df_filtrato = df[(df >= q_low) & (df <= q_high)]
    return df_filtrato


df_brutta = df_brutta.apply(lambda x: rimuovi_outliers(x, 5))  # Rimuove gli outliers oltre il 5%

df_brutta = df_brutta.dropna()

"""

"""
def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    column_cleaned = column[(column >= lower_bound) & (column <= upper_bound)]
    return column_cleaned

prova = remove_outliers(df_brutta)

prova = prova.dropna()
"""

"""
def colonne_adatte_trasformazione_log(data):
    colonne_adatte = []
    for colonna in data.columns:
        if data[colonna].dtype in [np.float64, np.int64]:
            skewness = data[colonna].skew()
            if skewness > 1 or skewness < -1:
                colonne_adatte.append(colonna)
    return colonne_adatte


def applica_trasformazione_log(data, colonne):
    for colonna in colonne:
        df[colonna + '_log'] = np.log(df[colonna])
    return data


# Identificazione delle colonne adatte e applicazione della trasformazione logaritmica
colonne_adatte = colonne_adatte_trasformazione_log(df_brutta)
df_brutta_trasformato = applica_trasformazione_log(df_brutta, colonne_adatte)
"""