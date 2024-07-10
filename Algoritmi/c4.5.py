import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Caricamento del dataset
data = pd.read_csv(r"C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\Algoritmi\online_news_CART.csv")

# Controlliamo la densità con un istogramma
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

#data['classe'] = pd.qcut(data['shares'], q=3, labels=[1,2,3])

# Separazione delle feature e dell'etichetta
X = data.drop('classe', axis=1) # tutte le colonne eccetto 'successo'
y = data['classe']  # la colonna 'successo'

# Usiamo la PCA per ridurre la dimendionalità
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

# Ora che sappiamo il numero di componenti che ci servono utilizziamo di nuovo la PCA
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X)

# Utilizzamo SMOTE per il problema del bilanciamento delle classi
smote = SMOTE()
X_sampled, y_sampled = smote.fit_resample(X_pca, y) 

# mettere feature names e target names

# Divisione del dataset in training e testing set
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=42)

# Inizializzazione del modello DecisionTreeClassifier

# Crea un modello di albero decisionale con l'entropia (opzione 2 gini)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train) # addestra il modello sui dati di addestramento

# Utilizza il modello addestrato per fare previsioni sui dati di test.
y_pred = clf.predict(X_test)

# Valutazione del modello
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred)) #capire questo passaggio

# 95% di acuracy con smote

# 89% di acuracy con i terzili

# Genera un report dettagliato delle prestazioni del modello per ciascuna classe (precisione, richiamo, F1-score)
""" print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
 """
# matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred) # calcola la matrice e mostra il numero di predizioni corrette e incorrette per ciascuna classe.
print("\nConfusion Matrix:\n", conf_matrix)


# visualizzare la matrice di confusione
target_names = data['classe'].unique().tolist()
 # crea una figura di dimensioni 10x7
plt.figure(figsize=(10, 7))
# crea una mappa di calore per la matrice di confusione con annotazioni e colori.
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Supponiamo di avere le seguenti variabili:
# y_true: le etichette vere
# y_scores: le probabilità predette dal modello per ciascuna classe
# classes: l'elenco delle classi (es. [1, 2, 3, 4])