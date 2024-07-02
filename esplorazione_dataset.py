import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r'OnlineNewsPopularity.csv')

print(df.shape) # 39644 righe x 61 colonne

pd.set_option('display.max_columns', None) # A causa delle 61 colonne, bisogna disabilitare il limite del display.
print(df.head()) # Campione prime 5 righe
# A primo impatto sembra nella norma, con la maggior parte delle colonne in valori continui.
# Si nota la presenza di numerose colonne con valori di 0 (il motivo principale è che va a rappresentare il False)

print(df.info())
# Come specificato dalla sorgente, confermiamo la mancanza di valori nulli in ogni colonna.
# Inoltre ad eccezione del url e dello share (nostro target) sono tutti di tipo float.
# La presenza di numerose colonne di natura booleana comporta ad un'alta frequenza dei valori 0.0 e 1.0

print(df.isnull().sum()) # Check

# Griglia di istogrammi che vanno a raffigurare la distribuzione dei valori per ogni colonna.
df.hist(figsize=(25, 22))
plt.show()
# Si notano diverse colonne con un solo valore.
# Potrebbe essere insolito per quanto riguarda le colonne delle keywords.
# Consigliato un approndimento del dominio in merito.

# DISCLAIMER:
# Le scale numeriche variano tra un grafico all'altro causando una possibile male interpretazione.
# Inoltre per i grafici con una sola barra, non va a significare che hanno un valore a testa. (causa: spazio insufficiente del display) 

# Controlliamo effettivamente se ci sono colonne con 1 solo valore unico.
print(df.nunique()) # Troppe colonne.

# Ciclo sulle colonne e stampo i conteggi.
for col in df.columns:
    print(f"{col} : {df[col].nunique()}")
# Ad eccezione delle colonne di True/false (conteggio = 2), la maggior parte delle colonne hanno valori regolari.
# Indagherei sul perché colonne di min e max hanno pochi valori unici.

# Voglio ottenere i nomi delle colonne con una distribuzione altamente sbilanciata
# Ho impostato lo sbilanciamento al 90% per prova
feature_sbilanciate = []
for col in df.columns:
    if df[col].value_counts(normalize = True).max() > 0.90:
        feature_sbilanciate.append(col)
# Nelle vecchie prove avevo impostato il 98 e 95 %, ma evidentmente le feature non sono così sbilanciate.
# Con 90 % ho ottenuto 4 nomi.
# DUBITO delle riuscita di questo for ---> Rivedere il ciclo for.
        
print(f"Possibili anomalie: {feature_sbilanciate}")

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

# Valori degli indicatori della label
print(df[col].describe())
# Sono 39644 valori (visto che non ci sono Nan).
# I valori variano da 1 a 843300.
# I valori sono MOLTO SBILANCIATI (si nota dall'istogramma degli shares)
# Quindi la media non è assolutamente consigliata da utilizzare come treshold per mappatura (dipendente dagli outlier)

# POSSIBILE SOLUZIONE: prendiamo in considerazione la MEDIANA (il quartile 50%), per ottenere due classi BILANCIATE.
# Così da optare per una CLASSIFICAZIONE BINARIA.
# Se si vuole effettuare una multiclasse, è necessaria una ricerca nel dominio.

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

    df = df[(df['shares'] > lower_bound) & (df['shares'] < upper_bound)]
    
    
CLASSIFICAZIONE BINARIA VS MULTICLASSE:
    
    Utilizzo della mediana come soglia per la classificazione binaria
    
    threshold = df['shares'].median()
    df['binary_label'] = (df['shares'] > threshold).astype(int)

"""