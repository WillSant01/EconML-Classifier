import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from prettytable import PrettyTable



df = pd.read_csv(r"C:\Users\AdamPezzutti\repo_github\EconML-Classifier\OnlineNewsPopularity.csv")
df_brutta = df.copy()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(df.shape) # 39644 righe x 61 colonne

pd.set_option('display.max_columns', None) # A causa delle 61 colonne, bisogna disabilitare il limite del display.
print(df.head()) # Campione prime 5 righe

# A primo impatto sembra nella norma, con la maggior parte delle colonne in valori continui.
# Si nota la presenza di numerose colonne con valori di 0 (il motivo principale è che vanno a rappresentare il False)

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
#VISUALIZZAZIONE GRAFICA DELLA DISTRIBUZIONE DI OGNI COLONNA

# Griglia di istogrammi che vanno a raffigurare la distribuzione dei valori per ogni colonna.

df_brutta.hist(figsize=(25, 22))
plt.show()

# Si notano diverse colonne con un solo valore.
# Potrebbe essere insolito per quanto riguarda le colonne delle keywords.

# DISCLAIMER:
# Le scale numeriche variano tra un grafico all'altro causando una possibile male interpretazione.
# Inoltre per i grafici con una sola barra, non va a significare che hanno un valore a testa. (causa: spazio insufficiente del display)
 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# CONTROLLO VALORI UNICI

print(df.nunique()) # Troppe colonne.

# Ciclo sulle colonne e stampo i conteggi.

for col in df.columns:
    print(f"{col} : {df[col].nunique()}")
    
# Ad eccezione delle colonne di True/false (conteggio = 2), la maggior parte delle colonne hanno valori regolari.

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Vorrei vedere se le feature sono linearmente indipendenti

# Qua ottengo la matrice di correlazione tra le colonne del dataframe escludendo la prima colonna (essendo l'unica colonna in stringhe)
matrice_correlazione = df.iloc[:, 1:].corr()

# Mi disinteresso delle mancanze di correlazione (valori intorno allo 0).
# Voglio visualizzare le variabili fortemente correlate (sia direttamente che inversamente)
filtro = (abs(matrice_correlazione) >= 0.7) & (abs(matrice_correlazione) != 1)

# Filtra le correlazioni.
correlazioni_strette = matrice_correlazione[filtro]

# Riduco le dimensioni del dataframe (da 60 a 30)
correlazioni_strette = correlazioni_strette.dropna(how = 'all', axis = 1).dropna(how = 'all')
# Stampa solo le correlazioni strette
print(correlazioni_strette)

# Grazie a due funzioni di numpy,
# posso estrarre dal mio dataframe (usata come matrice), la matrice triangolare superiore (valori che stanno sopra la diagonale principale).
triang_sup = np.triu(np.ones_like(correlazioni_strette, dtype = bool))


print(df.iloc[:,-1].describe())

value_label1 = df_brutta.iloc[:,-1]
value_label2 = value_label1[value_label1 > 3395.380184]

print(value_label2.describe())

value_label2 = value_label1[value_label1 > 10000.00]

print(value_label2.describe())

value_label2 = value_label1[value_label1 > 20000.00]

print(value_label2.describe())

descrizione_ultima = value_label2.describe()

print(descrizione_ultima[0] / len(df_brutta) * 100, " %")
# ciclo for dentro la funzione per applicarlo ad ogni feature, con l'obiettivo di stampare la % degli outliers.


prima_lung = len(value_label2)


value_label2.hist(by=None, ax=None, grid=True, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None, figsize=None, bins=10, backend=None, legend=False)

print(len(value_label2))
print(value_label2.describe())

# Rappresento grafico la matrice nascondendo la matrice triangolare superiore
plt.figure(figsize=(18, 15))
sns.heatmap(correlazioni_strette, mask=triang_sup, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, linecolor='grey')
plt.title('Matrice di Correlazione')
plt.show()
# Si nota che la metà delle features sono in stretta relazione con una (massimo due) feature.
# Inoltre si scopre che la metà delle feature sono ordinate in base alla loro correlazione a due a due.
# DISCLAIMER:
# Si nota che i valori sono un pò approssimati

print(df.describe())
# Valori degli indicatori statistici standard per ogni colonna.
value = df.iloc[:,4].value_counts(ascending = True, sort = True, normalize = True)


value_1 = df.iloc[:,4]
value_2 = value_1[value_1 > 0.608696]
"""
plt.figure(figsize=(8, 6))
value_2.plot(kind='bar', color='skyblue')
plt.title('Distribuzione dei valori nella colonna "a"')
plt.xlabel('Valori')
plt.ylabel('Frequenza Relativa')
plt.show()
"""
outlier = value_2[value_2 > 1]
print(outlier)
print(len(df_brutta))
df_brutta.drop(df_brutta[df_brutta.iloc[:,4] == 701.00].index, inplace = True)

print("dopo: ",len(df_brutta))


value_1 = df_brutta.iloc[:,4]
value_2 = value_1[value_1 > 0.608696]


outlier = value_2[value_2 >= 1]
print(outlier)

#df.drop(df[df['city'] == 'Chicago'].index)


plt.boxplot(value_2)
plt.title('Distribuzione dei valori nella colonna "unique_tokens"')
plt.xlabel('Valori')
plt.ylabel('Frequenza.')
plt.show()

print(value_2.describe())

# Valori degli indicatori della label

print(df[df.columns[23]].describe())

# Sono 39644 valori (visto che non ci sono Nan).

# I valori variano da 1 a 843300.

# I valori sono MOLTO SBILANCIATI (si nota dall'istogramma degli shares)

# Quindi la media non è assolutamente consigliata da utilizzare come treshold per mappatura (dipendente dagli outlier)

# POSSIBILE SOLUZIONE: prendiamo in considerazione la MEDIANA (il quartile 50%), per ottenere due classi BILANCIATE.

# Così da optare per una CLASSIFICAZIONE BINARIA.

# Se si vuole effettuare una multiclasse, è necessaria una ricerca nel dominio.

# ELIMINARE LE RIGHE CON ALMENO UN OUTLIER.

df_float = df_brutta.select_dtypes(include=['float'])

def gestisci_outlier(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    col[col < lower_bound] = lower_bound
    col[col > upper_bound] = upper_bound
    return col

# Applica la funzione a ciascuna colonna float

df_brutta[df_float.columns] = df[df_float.columns].apply(gestisci_outlier)

print(df)

df_brutta.hist(figsize=(25, 22))
plt.show()

df_brutta[df.columns[23]].hist(figsize=(25, 22))
plt.show()

df_brutta.info()

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
    # Calcoliamo lo z-score
    z_scores = np.abs((df[colonna] - df[colonna].mean()) / df[colonna].std())

    # Identifichiamo gli outliers
    outliers = df[colonna][z_scores > z_score_threshold]

    return outliers

# Inserisci il nome della colonna che vuoi visualizzare

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
#FUNZIONE PER VISUALIZZARE GRAFICAMENTE LA DISTRIBUZIONE DEI VALORI ALL'INTERNO DELLA COLONNA 'SHARES'

# seleziona gli articoli con massimo 10'000 shares

df_filtered = df[df['shares'] <= 10000]

# Visualizza le informazioni per la colonna degli 'shares' filtrati

shares_stats_filtered = df_filtered['shares'].describe()
print("Statistiche Descrittive delle Condivisioni (fino a 10000):")
print(shares_stats_filtered)

# Numero totale di articoli nel subset filtrato

total_articles_filtered = len(df_filtered)
print(f"Numero totale di articoli (fino a 10000 condivisioni): {total_articles_filtered}")

# Visualizza la distribuzione degli 'shares' con una curva gaussiana per il subset filtrato

plt.figure(figsize=(12, 6))
sns.histplot(df_filtered['shares'], kde=True)

# Imposta l'unità di misura delle asse x

plt.xticks(ticks=range(0, 10001, 1000),
           labels=[f'{i // 1000} mila' for i in range(0, 10001, 1000)])

plt.title(f'Distribuzione dgli Share degli Articoli (Fino a 10001, Totale articoli: {total_articles_filtered})')
plt.xlabel('Numero di Condivisioni')
plt.ylabel('Frequenza')
plt.show()

# Calcola l'IQR (Intervallo Interquartile) e identifica gli outliers nel subset filtrato

Q1 = df_filtered['shares'].quantile(0.25)
Q3 = df_filtered['shares'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Trova gli articoli virali (outliers) in base al numero di condivisioni nel subset filtrato
viral_articles_filtered = df_filtered[df_filtered['shares'] > upper_bound]
print("Numero di Articoli Virali (fino a 10000 condivisioni):", len(viral_articles_filtered))
print("Articoli Virali (fino a 10000 condivisioni):")
print(viral_articles_filtered[['url', 'shares']])


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

#FUNZIONE PER CAPIRE LA SKEWNESS E VISUALIZZARLA GRAFICAMENTE (da migliorare) 

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
    sns.histplot(data=df, x=colonna, kde=True, color='darkblue', stat='density')

    # Imposta il titolo del grafico
    plt.title(f"Distribuzione di {colonna} (Skewness: {skewness:.2f})")

    # Mostra il grafico
    plt.show()

# Carica il tuo DataFrame
df = pd.read_csv(r"C:\Users\AdamPezzutti\repo_github\EconML-Classifier\OnlineNewsPopularity.csv")

# Chiama la funzione
analizza_distribuzione(df)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE .displot

# Rimuovi spazi indesiderati dai nomi delle colonne
df.columns = df.columns.str.strip()

# Verifica che il nome della colonna sia corretto
print("Nomi delle colonne nel DataFrame dopo la rimozione degli spazi:", df.columns)

# Esegui il codice di visualizzazione con limiti asse x
sns.displot(data=df, x='shares', kind='kde')
plt.xlim(0, 10000)
plt.title('Distribuzione delle condivisioni')
plt.show()

sns.displot(data=df, x='shares', hue='is_weekend', kind='kde')
plt.xlim(0, 10000)
plt.title('Distribuzione delle condivisioni nei weekend e nei giorni feriali')
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#FUNZIONE DESCRIBE PER COLONNA

if ' shares' in df.columns:
    descrizione = df[' shares'].describe()
    print(descrizione)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
print(df[[' shares', ' kw_avg_avg']].corr())
    
    
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