import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

df = pd.read_csv(r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\OnlineNewsPopularity.csv')
df_brutta = df.copy()

print(f"Dimensioni righe x colonne: {df_brutta.shape}") # 39644 righe x 61 colonne

pd.set_option('display.max_columns', None) # A causa delle 61 colonne, bisogna disabilitare il limite del display.

print(df_brutta.head())
# Decidiamo di togliere:
# - La colonna degli URL (gli indici identificano l'articolo)
# - La colonna IS_WEEKEND (abbiamo già le colonne per i giorni della settimana -> ridondanza)
# - La colonna TIMEDELTA (inutile per il business goal)
nomi = df_brutta.columns.to_list()

# Rimozione della colonna URL, TIMEDELTA e IS_WEEKEND
df_brutta.drop([nomi[0], nomi[1], nomi[38]], axis=1, inplace=True)
nomi = df_brutta.columns.to_list()

# Ciclo sulle colonne e stampo i conteggi.
for col in df_brutta.columns:
    print(f"{col} : {df[col].nunique()}")
# Ad eccezione delle colonne di True/false (conteggio = 2), la maggior parte delle colonne hanno valori regolari.
# Indagheremo sul perché colonne di min e max hanno pochi valori unici.


# Vogliamo gestire le colonne di true/false accorpandole in un'unica con un range di valori (1 - n_colonne_binarie)
def gestione_binario(data, keyword):
    # Prendo i nomi delle colonne binarie che mi interessano
    colonne_da_accorpare = [col for col in data.columns if data[col].nunique() == 2 and keyword in col]
    # Estraggo dal dataframe
    matrice_da_accorpare = data[colonne_da_accorpare].copy()
    
    # Creo una series con la mappatura dei valori in base alla posizione
    colonna_nuova = pd.Series(matrice_da_accorpare.values.argmax(axis = 1) + 1)
    
    # Li tolgo dal df e aggiungo la colonna
    data.drop(columns = colonne_da_accorpare, inplace = True)
    data[colonna_nuova.name] = colonna_nuova
    
    data.rename(columns={None: keyword}, inplace=True)

    
    return data

# Per i topic
df_brutta = gestione_binario(df_brutta, "data_channel")
# Per i giorni della settimana
df_brutta = gestione_binario(df_brutta, "weekday")


def count_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers)

outliers_count = df_brutta.apply(count_outliers)


#print(outliers_count)

# Calcola la percentuale di outlier per ogni colonna
total_rows = df_brutta.shape[0]
outliers_percentage = (outliers_count / total_rows) * 100

# Aggiungo il simbolo '%'
outliers_percentage = outliers_percentage.map(lambda x: f'{x:.2f}%')

# Visualizzazione Risultati

outliers_summary = pd.DataFrame({
    'Numero di Outliers': outliers_count,
    'Percentuale di Outliers': outliers_percentage})

# Stampa il risultato
print(outliers_summary)

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

