# Progetto unito

Repository per la generazione di dati fittizzi relativi a transazioni bancarie.


## Istruzioni

In questo repository sono presenti due directory principali:
- **Dataset generator**: contiene il progetto per la generazione dei dataset

- **Preview dati**: contiene un file .csv che mostra:
  - composizione dei dati reali e distorsioni nei nomi e negli indirizzi dei gruppi di transazioni
  - distribuzione delle transazioni
  - distribuzione di nomi e indirizzi per ogni gruppo di transazioni

---

### Dataset generator
In Dataset generator sono presenti 2 file python:
- **dataset_generator.py**: consente di generare gruppi di transazioni. I dataset generati vengono salvati sotto la cartella **./output**

    - **sintassi**: *python3 dataset_generator.py [SHOW]*
        - con **[SHOW]** parametro opzionale, mostra in preview i dati generati

    - i parametri per la generazione dei dati sono definiti nel file **"config/parameters.json"**:
        - **"num_iban"**: imposta il numero di iban unici che vengono generati. Controlla il numero di entry nel dataset

        - **"min_range_entry"**: controlla il numero minimo di transazioni per ogni iban
        - **"max_range_entry"**: controlla il numero massimo di transazioni per ogni iban
        - **"min_range_holders"**: controlla il numero minimo di entità diverse associate ad ogni iban, se questi è condiviso. 
        - **"max_range_holders"**: controlla il numero massimo di entità diverse associate ad ogni iban, se questi è condiviso. (deve essere inferiore a max_range_entry)
        - **"V"**: fattore di variabilità. controlla il fattore di modifica delle parti societarie. (tra 0 e 1)
        - "EDIT":2,
        - **"T"**: Temperatura: controlla il fattore di distorsione dei nomi e degli indirizzi. (tra 0 e 1)
        - **"C"**: Changeable factor: controlla il fattore di aggiunta di spazi bianchi e  rimozione di parole nei nomi e negli indirizzi. (tra 0 e 1)


- **statistics.py**: Esegue delle semplici analisi statistiche sui dataset generati. I risultati di queste analisi vengono salvati sotto la cartella **./output_statistics**

    - **sintassi**: *python3 statistics.py [PATH]*
        - con [PATH]: path al dataset .xlsx
    
    - **DATASET_NAME_statistics.txt**: in output crea un file .txt in cui riassume:
        - preview dataset
        - distribuzione degli iban nelle transazioni
        - distribuzione di nomi e indirizzi in ogni gruppo di transazioni dello stesso iban

    - **distribuzione_iban_DATASET_NAME.png**: distribuzione degli iban per ogni gruppo di transazioni