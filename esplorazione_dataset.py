import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
import sys
from prettytable import PrettyTable


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
"""


script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
file_path = os.path.join(script_dir, "esplorazione_dataset.py")
print(os.getcwd())

df = pd.read_csv(r"C:\Users\AdamPezzutti\repo_github\EconML-Classifier\OnlineNewsPopularity.csv")
df_brutta = df.copy()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(df.shape) # 39644 righe x 61 colonne

pd.set_option('display.max_columns', None) # A causa delle 61 colonne, bisogna disabilitare il limite del display.
print(df.head()) # Campione prime 5 righe

# A primo impatto sembra nella norma, con la maggior parte delle colonne in valori continui.
# Si nota la presenza di numerose colonne con valori di 0 (il motivo principale è che vanno a rappresentare il False)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#CONTROLLA SE CI SONO DUPLICATI

#funzione di pandas per vedere se ci sono duplicati

duplicates = df[df.duplicated(keep=False)]

if not duplicates.empty:
    print(f"Sono state trovate {duplicates.shape[0]} righe duplicate:")
    print(duplicates)
else:
    print("Non ci sono righe duplicate nel dataset.")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#VISUALIZZAZIONE SOTTOFORMA DI TABELLA DELLE INFORMAZIONI PRINCIPALE X COLONNA

# Estrai le informazioni rilevanti dal DataFrame

info_df = pd.DataFrame({'Colonna': df.columns,
                        'Tipo Dati': df.dtypes,
                        'Valori Non Null': df.count(),
                        'Valori Null': df.isnull().sum()})

# Crea una tabella PrettyTable

table = PrettyTable()
table.field_names = info_df.columns.tolist()

# Aggiungi le righe della tabella

for index, row in info_df.iterrows():
    table.add_row(row.tolist())

# Stampa la tabella

print(table)

# Come specificato dalla sorgente, confermiamo la mancanza di valori nulli in ogni colonna.
# Inoltre ad eccezione del url e dello share (nostro target) sono tutti di tipo float.
# La presenza di numerose colonne di natura booleana comporta ad un'alta frequenza dei valori 0.0 e 1.0

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#VISUALIZZARE VALORI NULLI

print(df.isnull().sum()) # Check

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# CONTROLLO VALORI UNICI

print(df.nunique()) # Troppe colonne.

# Ciclo sulle colonne e stampo i conteggi.

for col in df.columns:
    print(f"{col} : {df[col].nunique()}")
    
# Ad eccezione delle colonne di True/false (conteggio = 2), la maggior parte delle colonne hanno valori regolari.

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Elimina gli spazi davanti ai nomi delle colonne

df.columns = df.columns.str.strip()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE DESCRIBE PER COLONNA

def describe_column(df, column_name):
    
    # Controlla se il nome della colonna esiste nel DataFrame
  
    if column_name in df.columns:
        descrizione = df[column_name].describe()
        return descrizione
    else:
        print(f"La colonna '{column_name}' non esiste nel DataFrame.")
        return None

column_name = input("Inserisci il nome della colonna da descrivere:")
descrizione = describe_column(df, column_name)

if descrizione is not None:
    print(descrizione)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE PER VEDERE LA CORRELAZIONE TRA DUE COLONNE

def calcolo_correlazione(df, col1, col2):
    
   #fa la matrice di correlazione, e con iloc seleziona la colonna
   
    correlazione = df[[col1, col2]].corr().iloc[0, 1]
    return correlazione

col1 = input("Inserisci il nome della prima colonna: ")
col2 = input("Inserisci il nome della seconda colonna: ")
corr_value = calcolo_correlazione (df, col1, col2)

print(f"La correlazione tra {col1} e {col2} è: {corr_value}")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE PER CAPIRE LA SKEWNESS E VISUALIZZARLA GRAFICAMENTE (distribuzione dei dati) 

def analizza_distribuzione(df):
    # Chiedi all'utente il nome della colonna
    colonna = input("Inserisci il nome della colonna da analizzare: ")

    # Controlla se la colonna esiste nel DataFrame
    if colonna not in df.columns:
        print(f"La colonna '{colonna}' non esiste nel DataFrame.")
        return

    # Calcola la skewness
    skewness = df[colonna].skew()
    print(f"La skewness della colonna '{colonna}' è: {skewness:.2f}")

    # Interpretazione della skewness
    if skewness > 0:
        print(f"La distribuzione della colonna '{colonna}' è positivamente skewed (asimmetrica a destra).")
    elif skewness < 0:
        print(f"La distribuzione della colonna '{colonna}' è negativamente skewed (asimmetrica a sinistra).")
    else:
        print(f"La distribuzione della colonna '{colonna}' è approssimativamente simmetrica.")

    # Crea l'istogramma con curva di densità
    sns.histplot(data=df, x=colonna, color='darkblue', stat='count')

    # Imposta il titolo del grafico
    plt.title(f"Distribuzione di {colonna} (Skewness: {skewness:.2f})")

    # Se la colonna è 'shares', limita l'asse X a 10000
    if colonna == 'shares':
        plt.xlim(0, 10000)
        plt.xlabel('Numero di condivisioni (limite a 10000)')

    # Mostra il grafico
    plt.show()

# Chiama la funzione
analizza_distribuzione(df)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE PER CAPIRE E VISUALIZZARE LE COLONNE ALTAMENTE SBILANCIATE

#impostO lo sbilanciamento al 90% per prova

def find_imbalanced_features(df, threshold=0.9):

  imbalanced_features = []
  for col in df.columns:
    if df[col].value_counts(normalize=True).max() > threshold:
      imbalanced_features.append(col)
  return imbalanced_features

# Trova le feature sbilanciate

features_sbilanciate = find_imbalanced_features(df, 0.9)

# Visualizza la distribuzione delle feature sbilanciate

for feature in features_sbilanciate:
  plt.figure()
  plt.title(f"Distribuzione di {feature}")
  df[feature].value_counts().plot(kind='bar')
  plt.xlabel("Valore")
  plt.ylabel("Conteggio")
  plt.show()

print(f"Possibili anomalie: {features_sbilanciate}")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#VISUALIZZAZIONE GRAFICA DELLA DISTRIBUZIONE DI OGNI COLONNA (nome con spazio)

# Griglia di istogrammi che vanno a raffigurare la distribuzione dei valori per ogni colonna.

def plot_column_distribution(df, column_name):

    if column_name in df.columns:
        df[column_name].hist(figsize=(10, 6))
        plt.title(f'Distribuzione della colonna: {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequenza')
        plt.show()
    else:
        print(f"La colonna '{column_name}' non esiste nel DataFrame.")

column_name = input("Inserisci il nome della colonna da visualizzare:")
plot_column_distribution(df_brutta, column_name)

# Si notano diverse colonne con un solo valore.
# Potrebbe essere insolito per quanto riguarda le colonne delle keywords.

# DISCLAIMER:
# Le scale numeriche variano tra un grafico all'altro causando una possibile male interpretazione.
# Inoltre per i grafici con una sola barra, non va a significare che hanno un valore a testa. (causa: spazio insufficiente del display)
 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE PER VISUALIZZARE GRAFICAMENTE LA DISTRIBUZIONE DEI VALORI ALL'INTERNO DELLA COLONNA 'SHARES'(la nostra label)

# seleziono gli articoli con massimo 10'000 shares

df_filtered = df[df['shares'] <= 10000]

# Visualizzo le informazioni per la colonna degli 'shares' filtrati

shares_stats_filtered = df_filtered['shares'].describe()
print("Statistiche Descrittive delle Condivisioni (fino a 10000):")
print(shares_stats_filtered)

# Numero totale di articoli nel subset filtrato

total_articles_filtered = len(df_filtered)
print(f"Numero totale di articoli (fino a 10000 condivisioni): {total_articles_filtered}")

# Visualizzo la distribuzione degli 'shares' con una curva gaussiana per il subset filtrato

plt.figure(figsize=(12, 6))
sns.histplot(df_filtered['shares'], kde=True)

# Imposto l'unità di misura delle asse x

plt.xticks(ticks=range(0, 10001, 1000),
           labels=[f'{i // 1000} mila' for i in range(0, 10001, 1000)])

plt.title(f'Distribuzione degli Share degli Articoli (Fino a 10001, Totale articoli: {total_articles_filtered})')
plt.xlabel('Numero di Condivisioni')
plt.ylabel('Frequenza')
plt.show()

# Calcolo l'IQR (Intervallo Interquartile) e identifico gli outliers nel subset filtrato

Q1 = df_filtered['shares'].quantile(0.25)
Q3 = df_filtered['shares'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Trovo gli articoli virali (outliers) in base al numero di condivisioni nel subset filtrato
articles_filtered = df_filtered[df_filtered['shares'] > upper_bound]
print("Numero di Articoli filtrati (fino a 10000 condivisioni):", len(articles_filtered))
print("Articoli filtrati (fino a 10000 condivisioni):")
print(articles_filtered[['url', 'shares']])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE PER VEDERE NUMERICAMENTE IL NUMERO E LA % DI OUTLIERS PER OGNI COLONNA

# Escludi la prima colonna non numerica

df_numeric = df.iloc[:, 1:]

# Funzione per calcolare il numero di outlier in una colonna

def count_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers)

# Calcola il numero di outlier per ogni colonna numerica

outliers_count = df_numeric.apply(count_outliers)

# Calcola la percentuale di outlier per ogni colonna

total_rows = df_numeric.shape[0]

outliers_percentage = (outliers_count / total_rows) * 100

# Aggiungo il simbolo '%'

outliers_percentage = outliers_percentage.map(lambda x: f'{x:.2f}%')

# Visualizzazione Risultati

outliers_summary = pd.DataFrame({
    'Numero di Outliers': outliers_count,
    'Percentuale di Outliers': outliers_percentage})

# Stampa il risultato

print(outliers_summary)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE PER VISUALIZZARE ATTRAVERSO UN BOX PLOT GLI OUTLIERS

# Definiamo una funzione per identificare gli outliers

def trova_outliers(df, colonna, z_score_threshold=3):
    # Calcolo lo z-score
    z_scores = np.abs((df[colonna] - df[colonna].mean()) / df[colonna].std())

    # Identifico gli outliers
    outliers = df[colonna][z_scores > z_score_threshold]

    return outliers

# inserire in input la colonna da analizzare

colonna_da_analizzare = input("Inserisci il nome della colonna da analizzare: ")

# Verifica che la colonna esista e sia numerica

if colonna_da_analizzare in df.columns and df[colonna_da_analizzare].dtype in [np.int64, np.float64]:
    
    # Identifichiamo gli outliers per la colonna specificata
    outliers = trova_outliers(df, colonna_da_analizzare)

    # Visualizziamo gli outliers
    print(f"Valori degli outliers in {colonna_da_analizzare}:")
    print(outliers)

    # Creiamo un box plot per la colonna specificata
    plt.figure()
    plt.boxplot(df[colonna_da_analizzare].dropna())  # Rimuoviamo i valori NaN per evitare problemi nel plotting
    plt.title(f'Box plot di {colonna_da_analizzare}')
    plt.xlabel(colonna_da_analizzare)
    plt.ylabel('Valori')
    plt.show()
else:
    print(f"La colonna '{colonna_da_analizzare}' non esiste nel DataFrame o non è numerica.")
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE PER DIVIDERE IL DATASET IN TRE CLASSI

# definiamo le soglie per le tre classi

soglia_bassa = 900

soglia_alta = 4250

# funzione def per creare le classi

def classificazione_shares(shares):
    if shares < soglia_bassa:
        return 'Poco Popolare'
    elif shares < soglia_alta:
        return 'Moderatamente Popolare'
    else:
        return 'Molto Popolare'

# Applica la funzione di classificazione al DataFrame

df['classe'] = df['shares'].apply(classificazione_shares)

# Conta il numero di articoli per ciascuna classe e crea un DataFrame

conteggio_classi = df['classe'].value_counts().reset_index()

conteggio_classi.columns = ['Classe', 'Conteggio']

# Visualizza la tabella nella console con il numero preciso di articoli 

print(conteggio_classi)

# Crea un grafico a barre per la visualizzazione grafica delle classi

plt.bar(conteggio_classi['Classe'], conteggio_classi['Conteggio'])
plt.xlabel('Classe')
plt.ylabel('Numero di articoli')
plt.title('Distribuzione delle classi')
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Business Goal: Predirre la popolarità mediatica di 
vari articoli di Mashable,
prendendo in considerazione diverse metriche derivanti da
una indagine statistica.
"""


"""
GESTIONE DELLE COLONNE CON VALORI UNICI:
    
    Elimina le colonne con un solo valore unico

    df = df.loc[:, df.nunique() > 1]


FEATURE ENGINEERING:
    
    Esempio di creazione di una nuova feature combinata

    df['nuova_feature'] = df['feature1'] + df['feature2']
    
SELEZIONE DELLE FEATURE:
    
    from sklearn.feature_selection import SelectKBest, f_classif

    X = df.drop(['target_column'], axis=1)
    y = df['target_column']

    Seleziona le migliori 10 feature utilizzando il test ANOVA F-value
    
    selector = SelectKBest(score_func=f_classif, k=10)
    X_new = selector.fit_transform(X, y)
    
    
GESTIONE DEGLI OUTILIER:
    
    from scipy import stats

    Identifica e gestisci gli outlier nella colonna 'shares' utilizzando il metodo IQR
    
    Q1 = df['shares'].quantile(0.25)
    Q3 = df['shares'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_brutta = df_brutta[(df_brutta > lower_bound) & (df_brutta < upper_bound)]
    
    
CLASSIFICAZIONE BINARIA VS MULTICLASSE:
    
    Utilizzo della mediana come soglia per la classificazione binaria
    
    threshold = df['shares'].median()
    df['binary_label'] = (df['shares'] > threshold).astype(int)

"""