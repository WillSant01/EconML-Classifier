import pandas as pd
import numpy as np
from scipy.stats import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(
    r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\OnlineNewsPopularity.csv')
df.columns = df.columns.str.lstrip()  # tolgo gli spazi negli indici
df_brutta = df.copy()

df_brutta.drop(['url', 'timedelta', 'is_weekend', 'abs_title_subjectivity',
               'abs_title_sentiment_polarity'], axis=1, inplace=True)


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

X = df_brutta.drop(['shares'], axis=1)
y = df_brutta['shares']

sns.histplot(y, kde=True)
plt.title('Distribuzione della label (shares)')
plt.xlabel('Valore della label')
plt.ylabel('Frequenza')
plt.show()

y_log = np.log1p(y)
# y_boxcox, _ = boxcox(y + 1)

sns.histplot(y_log, kde=True)
plt.title('Distribuzione della label (shares)')
plt.xlabel('Valore della label')
plt.ylabel('Frequenza')
plt.show()

y_binned = pd.qcut(y_log, q=3, labels=[1, 2, 3])

#scaler = StandardScaler()
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calcola la varianza spiegata (valore degli autovalori per ogni numero di componenti)
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
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_binned, test_size=0.3, random_state=42)

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(
), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

cv_scores = cross_val_score(best_model, X_train, y_train, cv = 5, scoring = 'accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean()}")

y_pred = best_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))