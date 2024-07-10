import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

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


# Divisione del dataset in training e testing set
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Addestramento del modello di regressione logistica
model = LogisticRegression()
model.fit(X_train, y_train)


# Predizione sui dati di test
y_pred = model.predict(X_test)

# Valutazione dell'accuratezza del modello
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)