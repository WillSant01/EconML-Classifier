import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Caricamento del dataset
data = pd.read_csv(r"C:\Users\FrancescoFenzi\repo_git\EconML-Classifier\Algoritmi\online_news_outliers.csv")

# Separazione delle feature e dell'etichetta
X = data.drop('classe', axis=1)  # tutte le colonne eccetto 'classe'
y = data['classe']  # la colonna 'classe'

# Divisione del dataset in training e testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inizializzazione del modello SVC (Support Vector Classifier)
svc = SVC(random_state=42)

# Valutazione del modello iniziale
svc.fit(X_train, y_train)
y_pred_initial = svc.predict(X_test)
print(f'Initial Accuracy: {accuracy_score(y_test, y_pred_initial)}')
print(classification_report(y_test, y_pred_initial))

# 62% di accuracy senza ottimizazione

# Tuning (ottimizzazione)

# Definizione della griglia di iperparametri
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

""" C : parametro di regolarizzazione. 
    Valori più alti di C significano meno regolarizzazione, 
    cioè il modello cerca di classificare correttamente tutti i punti di addestramento, 
    mentre valori più bassi di C consentono qualche errore per migliorare la generalizzazione. 
    
    gamma : Parametro del kernel che definisce l'influenza di un singolo punto di addestramento. 
    È importante per i kernel non lineari.
    
    kernel : tipo di kernel da utilizzare nell'algoritmo
"""

# Inizializzazione del GridSearchCV
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

""" GridSearchCV : È uno strumento di scikit-learn che permette di effettuare una ricerca attraverso una griglia specificata di parametri per un modello.

    estimator : Il modello che si vuole ottimizzare, in questo caso un SVC (Support Vector Classifier).
    param_grid : La griglia di iperparametri definita in precedenza.
    
    cv : Cross-validation, il numero di suddivisioni del dataset da usare per la validazione incrociata. 
    Qui, cv=5 indica che si utilizzerà una cross-validation a 5 fold.
    scoring: La metrica di valutazione da ottimizzare. Qui viene utilizzata l'accuratezza (accuracy).
   
    n_jobs: Il numero di job da eseguire in parallelo. -1 significa che verranno utilizzati tutti i processori disponibili.
"""

# Predizione sui dati di test con il miglior modello
best_svc = grid_search.best_estimator_
y_pred = best_svc.predict(X_test)

""" 
grid_search.best_estimator_: Una volta completata la ricerca sulla griglia, 
GridSearchCV fornisce il miglior modello trovato (best_estimator_), cioè il modello con la combinazione di iperparametri che ha ottenuto le migliori prestazioni durante la cross-validation.

best_svc : Questo è il miglior modello SVC trovato.

y_pred = best_svc.predict(X_test) : Utilizza il miglior modello per fare predizioni sui dati di test (X_test), 
ottenendo così le predizioni (y_pred).
"""

# Valutazione del modello
print(f'Best parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))