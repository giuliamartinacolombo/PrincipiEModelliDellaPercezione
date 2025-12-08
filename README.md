# 📌 Descrizione 
Questo progetto, sviluppato per l’insegnamento Principi e Modelli della Percezione (anno scolastico 2025/2026), ha come obiettivo l’analisi e la visualizzazione della struttura dei cluster presenti in diversi dataset utilizzando l’algoritmo OPTICS (Ordering Points To Identify the Clustering Structure).

Il lavoro comprende:
- utilizzo di tecniche di clustering non supervisionato,
- applicazione dell’algoritmo OPTICS tramite Python,
- visualizzazione del reachability plot e dei cluster risultanti,
- confronto con altri algoritmi basati sulla densità (es. DBSCAN),
- analisi e interpretazione dei risultati.

## Struttura della repository

```
application/
    penguins_optics.py            # script principale con OPTICS e visualizzazioni
    relazione_esperimento.md      # relazione della parte pratica
    penguins.csv                  # dataset

    images/
        PCA_sex_MALE.png          # PCA 2D con sesso incluso
        PCA_no_sex.png            # PCA 2D senza sesso
        PCA_otherXi               # PCA 2D con la variazione del parametro Xi
        Reachability_sex_MALE.png # reachability plot con sesso
        Reachability_no_sex.png   # reachability plot senza sesso
        heatmap.png
        histogram.png
    
docs/                       
    bibliografia.md               # riferimenti bibliografici
    relazione_progetto.md         # relazione teorica su clustering e OPTICS
    slide_OPTICS.pdf              # slide di presentazione del progetto

README.md                     # questo file
```

## Requisiti

Per eseguire il codice è fondamentale che:
* Python sia installato sulla macchina
* I file ```penguins_optics.py``` e ```penguins.csv``` siano nella stessa cartella
* Le seguenti librerie di Python siano installate:
  * pandas
  * numpy
  * matplotlib
  * scikit-learn

Installazione rapida:
```
pip install pandas numpy matplotlib scikit-learn

```
## Documentazione

La documentazione si divide in due parti:
* ```docs/relazione_progetto.md``` -> relazione sulla parte teorica
* ```application/relazione_esperimento.md``` -> relazione sulla parte pratica (applicazione di OPTICS e analisi dei risultati)

  

