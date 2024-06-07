# UnitoER_ISP_Test

Passi per lanciare i test:
- installare i requirements: **pip install -r requirements.txt**
- eseguire da riga di comando: **python3 unitoBert.py PATH_FILE**
  - con **PATH_FILE**: il path al dataset

Alcuni dataset sono già disponibili sotto la directory *./dataset*. una volta lanciato un test, il programma:
1. stampa a video i risultati di predizione: (accuracy, precision, recall, f1score)
2. salva i risultati in un file di log .txt sotto la cartella *./Test*
3. salva un plot sotto la cartella *./Plot