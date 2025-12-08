# 📌 Descrizione 
Questo progetto, sviluppato per l’insegnamento Principi e Modelli della Percezione (anno scolastico 2025/2026), ha come obiettivo l’analisi e la visualizzazione della struttura dei cluster presenti in diversi dataset utilizzando l’algoritmo OPTICS (Ordering Points To Identify the Clustering Structure).

Il lavoro comprende:
- utilizzo di tecniche di clustering non supervisionato,
- applicazione dell’algoritmo OPTICS tramite Python,
- visualizzazione del reachability plot e dei cluster risultanti,
- confronto con altri algoritmi basati sulla densità (es. DBSCAN),
- analisi e interpretazione dei risultati.

## Struttura della repository

```text
application/
    penguins_optics.py            # script principale con OPTICS e visualizzazioni
    relazione_esperimento.md      # relazione della parte pratica 

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
