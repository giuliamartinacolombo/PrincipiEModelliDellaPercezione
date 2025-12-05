# Applicazione di OPTICS – Dataset “Penguins” #

introduzione

---

### **1. SETTING DELL'AMBIENTE** ###
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
  * ```culmen_length_mm```, ```culmen_depth_mm``` , ```flipper_length_mm``` , ```body_mass_g``` (di tipo float64)
  * ```sex``` (di tipo object)

Viene inoltre evidenziata la presenza di valori mancanti.

---

### **2. PREPROCESSING DEI DATI** ###
Prima di poter applicare l’algoritmo è stato necessario ripulire e trasformare i dati.

<br>

#### **Rimozione dei valori mancanti** ####

```
df = df.dropna()
```
 
Si eliminano le righe contenenti almeno un valore mancante; questo riduce leggermente il numero di pinguini ma garantisce che l’algoritmo lavori su un dataset completo.

<br>

#### **Codifica della variabile categorica *sex*** ####

OPTICS richiede esclusivamente feature numeriche. La variabile sex è stata trasformata in una variabile binaria:

```
df = pd.get_dummies(df, columns=["sex"], drop_first=True)  
print(df.head())
``` 
L’opzione ```drop_first=True``` evita la collinearità: invece di creare due colonne (sex_FEMALE, sex_MALE) ridondanti, viene mantenuta solo ```sex_MALE``` (0 = femmina, 1= maschio)

Le prime righe risultano, ad esempio:
```
    culmen_length   culmen_depth    flipper_length  	body_mass  sex_MALE
0          	39.1         	18.7          	181.0   	3750.0     	1
1          	39.5         	17.4          	186.0   	3800.0    	0
2          	40.3         	18.0          	195.0   	3250.0    	0
4          	36.7         	19.3          	193.0   	3450.0    	0
5          	39.3         	20.6          	190.0   	3650.0   	1
```
---
 
### **3 – SELEZIONE DELLE FEATURE** ###
 
Sono state selezionate tutte le feature disponibili per il clustering:

-   ```culmen_length_mm``` -> Lunghezza del becco
-   ```culmen_depth_mm``` -> Profondità del becco
-   ```flipper_length_mm``` -> Lunghezza della pinna
-   ```body_mass_g``` -> Massa corporea
-   ```sex_MALE``` -> Sesso
     
``` 
features = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
	“sex_male” #flag che indica il sesso
]
 
X = df[features].values
```
```X``` è quindi una matrice di dimensione (*n_pinguini, 5*)

---

### **4 – STANDARDIZZAZIONE DELLE FEATURE** ###
 
Le variabili presentano scale molto diverse (millimetri, grammi, variabile binaria) che non sarebbero paragonabili. Poiché OPTICS si basa sulle distanze, è necessario riportarle su una scala comparabile tramite standardizzazione:
 
```
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #applica ad ogni colonna una formula 
 
 
print("Shape X_scaled:", X_scaled.shape) 
print("Prima riga:", X_scaled[0]) 
```

L’output conferma che:

* ```X_scale```d ha dimensione (335, 5)

* ogni riga rappresenta un pinguino in uno spazio a 5 dimensioni standardizzato, ad esempio:

```Prima riga: [-0.89772327  0.77726336 -0.12689335 -0.57223347  0.99108452]```
