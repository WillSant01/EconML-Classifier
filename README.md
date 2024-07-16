# Dataset per la previsione della Popolarità degli Articoli di Notizie Online

DESCRIZIONE

    Questo dataset contiene informazioni su articoli di notizie online pubblicati da Mashable. Non include il contenuto originale degli articoli, ma bensì dati statistici associati ad essi, come il numero di condivisioni, i like e i commenti. Lo scopo del dataset è quello di aiutare a sviluppare modelli di machine learning per prevedere la popolarità degli articoli di notizie online.

INFORMAZIONI SULLA FONTE:

    - Creatori: Kelwin Fernandes, Pedro Vinagre, Pedro Sernadela
    - Data: Maggio 2015
    - Donatore: Kelwin Fernandes


UTILIZZO:

	Questo dataset è stato utilizzato per un compito di classificazione multiclasse per prevedere la popolarità degli articoli come "non popolare", "poco popolare", "mediamente popolare" e "molto popolare". Sono stati utilizzati gli algoritmi CART 4.5 e Random Forest, ottenendo un'accuratezza rispettivamente del 90% e del 94%.
	

DETTAGLI DEL DATASET:

| Caratteristica | Valore |
|---|---|
| Numero di Istanze | 39.797 |
| Numero di Attributi | 61 (58 predittivi, 2 non predittivi, 1 campo obiettivo) |
| Valori Mancanti degli Attributi | Nessuno |
| Distribuzione delle classi: |
| | Valore della classe (condivisioni) | Soglia multiclasse |
| | --- | --- |
| | 0 - 900) | Non popolare |
| | [900 - 1500) | Poco popolare |
| | [1500 - 3000) | Popolare |
| | 3000 + | Molto popolare |



REQUISITI

    ° Python 3.x;

        - Per verificare la versione di Python, esegui il seguente comando:

            python --version
	

INSTALLAZIONE DELLE DIPENDENZE

	Le librerie richieste per il progetto sono elencate nel file requirements.txt. Assicurati di avere installato il kernel dell'editor che stai utilizzando, oltre alle librerie elencate nel file requirements.txt. Per installare le librerie, esegui il seguente comando:

    pip install -r requirements.txt


AVVIO DEL PROGETTO

    Segui questi passaggi per avviare tutti i processi della classificazione:

    - Clona il repository git che contiene i codici.
    - Assicurati di avere installato tutte le dipendenze elencate nel file requirements.txt.

        pip install -r requirements.txt

    - Esegui i codici algoritmi/c4.5.py e algoritmi/random_forest.py per effettuare le classificazioni.

CODICI PYTHON:

    Il repository include diversi file Python:

    ° Esplorazione dati:
        - esplorazione_dataset.py: Questo script esplora il dataset e fornisce statistiche descrittive;
        - OnlineNewsPopularity,csv: Questo file CSV contiene i dati originali del dataset.

    ° Pre-Processing:
        - normalizzazione_dataset.py: Questo script normalizza i dati e crea due nuovi file CSV: online_news_CART.csv e online_news_RF.csv.

    ° Classificazione:
        - c4.5.py: Questo script implementa l'algoritmo CART 4.5 per la classificazione della popolarità degli articoli di notizie;
        - random_forest.py: Questo script implementa l'algoritmo Random Forest per la classificazione della popolarità degli articoli di notizie.


CITAZIONI:

   K. Fernandes, P. Vinagre e P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.


Contatti:

    Per domande o informazioni:

    - william.santeramo@itsrizzoli.it
    - francesco.fenzi@itsrizzoli.it
    - adam.pezzutti@itsrizzoli.it
    - piero.tacunan@itsrizzoli.it
    - denny.rutigliano@itsrizzoli.it