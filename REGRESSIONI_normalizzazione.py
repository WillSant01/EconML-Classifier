import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.decomposition import PCA 

df = pd.read_csv(r'C:\Users\WilliamSanteramo\Repo_github\EconML-Classifier\OnlineNewsPopularity.csv')
df.columns = df.columns.str.lstrip()
df_brutta = df.copy()

df_brutta.drop(['url', 'timedelta', 'is_weekend'], axis=1, inplace=True)

df_brutta["kw_min"] = df_brutta[["kw_min_min", "kw_min_max", "kw_min_avg"]].mean(axis=1)
df_brutta["kw_max"] = df_brutta[["kw_max_min", "kw_max_max", "kw_max_avg"]].mean(axis=1)
df_brutta["kw_avg"] = df_brutta[["kw_avg_min", "kw_avg_max", "kw_avg_avg"]].mean(axis=1)

df_brutta.drop(["kw_min_min", "kw_min_max", "kw_min_avg", "kw_max_min", "kw_max_max", "kw_max_avg"], axis=1, inplace=True)

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

def clip_outliers(data, lower_quantile = 0.01, upper_quantile = 0.99):
    quantili = data.quantile([lower_quantile, upper_quantile])
    
    for col in data.columns:
        data[col] = np.clip(data[col], quantili.loc[lower_quantile, col], quantili.loc[upper_quantile, col])
    return data

df_brutta = clip_outliers(df_brutta)

scaler = StandardScaler()

X = df_brutta.drop(columns = ['shares'])
y = df_brutta['shares']
X_scaled = scaler.fit_transform(X)

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

pca = PCA(n_components=32)
X_pca = pca.fit_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)

print("Mean squared error: ", mean_squared_error(y_test, y_pred_lin))
print("R^2 Score: ", r2_score(y_test, y_pred_lin))


def classificazione_shares(shares):
    if shares < 3000:
        return 1
    else:
        return 2
    
df_brutta['classe'] = df_brutta['shares'].apply(classificazione_shares)
df_brutta.drop(columns="shares", inplace=True)


y_bin = df_brutta['classe']

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_bin, test_size = 0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))