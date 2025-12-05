# Applicazione di OPTICS – Dataset “Penguins” #

introduzione

#### **1. SETTING DELL'AMBIENTE**
Per prima cosa sono state importate tutte le librerie necessarie al funzionamento del programma:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #per i grafici
 
from sklearn.preprocessing import StandardScaler #per la standardizzazione delle feature
from sklearn.cluster import OPTICS #algoritmo di clustering
from sklearn.decomposition import PCA #per la riduzione della dimensionalità
 ```
Il dataset è stato caricato da file CSV in un DataFrame pandas e ne è stata effettuata una prima ispezione:

```
df = pd.read_csv(file_path) 
print(df.head())
print(df.info())
```
L’output mostra le prime righe e alcune informazioni strutturali:
- 344 righe totali
- 5 colonne:
  * culmen_length_mm, culmen_depth_mm , flipper_length_mm , body_mass_g (di tipo float64)
  * sex (di tipo object)

Viene inoltre evidenziata la presenza di valori mancanti.

