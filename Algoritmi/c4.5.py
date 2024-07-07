import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Caricamento del dataset
data = pd.read_csv('OnlineNewsPopularity.csv')

# Separazione delle feature e dell'etichetta
X = data.drop('successo', axis=1) # tutte le colonne eccetto 'successo'
y = data['successo']  # la colonna 'successo'

# mettere feature names e target names

# Divisione del dataset in training e testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inizializzazione del modello DecisionTreeClassifier

# Crea un modello di albero decisionale con l'entropia (opzione 2 gini)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train) # addestra il modello sui dati di addestramento

# Utilizza il modello addestrato per fare previsioni sui dati di test.
y_pred = clf.predict(X_test)

# Valutazione del modello
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred)) #capire questo passaggio

# Genera un report dettagliato delle prestazioni del modello per ciascuna classe (precisione, richiamo, F1-score)
""" print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
 """
# matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred) # calcola la matrice e mostra il numero di predizioni corrette e incorrette per ciascuna classe.
print("\nConfusion Matrix:\n", conf_matrix)

# Visualizzazione dell'albero decisionale
""" plt.figure(figsize=(20, 10)) # crea una figura di dimensioni 20x10
plot_tree(clf, feature_names=feature_names, class_names=X, filled=True) # visualizza l'albero decisionale con nomi delle caratteristiche e delle classi
plt.title("Decision Tree - C4.5 (Wine Dataset)")
plt.show() """

# visualizzare la matrice di confusione

""" # crea una figura di dimensioni 10x7
plt.figure(figsize=(10, 7))
# crea una mappa di calore per la matrice di confusione con annotazioni e colori.
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show() """

# Tuning (ottimizzare)

# inizializzazione del modello DecisionTreeClassifier
clf = DecisionTreeClassifier()

# inizializzazione iperparametri
param_grid = {
    'max_depth': [3, 5, 10, None], # max_depth: profondità massima dell'albero
    'min_samples_split': [2, 5, 10], # min_samples_split: numero minimo di campioni richiesti per suddividere un nodo
    'min_samples_leaf': [1, 2, 4], # min_sample_leaf : numero minimo di campioni che un nodo foglia deve contenere
    'max_features': ['auto', 'sqrt', 'log2'] # max_features : numero massimo di caratteristiche da considerare per trovare la migliore divisione
}
# capire i parametri perfetti di param_grid

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

""" GridSearchCV : È uno strumento di scikit-learn che permette di effettuare una ricerca attraverso una griglia specificata di parametri per un modello.

    estimator : Il modello che si vuole ottimizzare, in questo caso un SVC (Support Vector Classifier).
    param_grid : La griglia di iperparametri definita in precedenza.
    
    cv : Cross-validation, il numero di suddivisioni del dataset da usare per la validazione incrociata. 
    Qui, cv=5 indica che si utilizzerà una cross-validation a 5 fold.
    scoring: La metrica di valutazione da ottimizzare. Qui viene utilizzata l'accuratezza (accuracy).
   
    n_jobs: Il numero di job da eseguire in parallelo. -1 significa che verranno utilizzati tutti i processori disponibili.
"""

# predizione sui dati di test con il miglior modello
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# valutazione modello con tuning
print(f'Best parameters found: , {grid_search.best_params_}')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))