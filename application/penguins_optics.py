import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #per i grafici

from sklearn.preprocessing import StandardScaler #per la standardizzazione delle feature
from sklearn.cluster import OPTICS  #algoritmo di clustering
from sklearn.decomposition import PCA #per la riduzione della dimensionalità


# ======== Lettura dataset ========

file_path = "penguins.csv" 

# Lettura del dataset e stampa delle prime 5 righe + informazioni inerenti il dataset
df = pd.read_csv(file_path) #legge in un dataframe il file .csv
print(df.head())
print(df.info())            #mostra informazioni sul dataset, come il numero di righe, colonne e tipi di dati, tra cui eventuali valori nulli

# drop di valori NA presenti nel dataset, a costo di perdere qualche pinguino
df = df.dropna()

# sex è una variabile categorica (MALE, FEMALE, NA), va convertita in numerica perché OPTICS lavora solo con variabili numeriche
df = pd.get_dummies(df, columns=["sex"], drop_first=True)  #evito collinearità (get dummies crea una colonna binaria), ossia evito di avere sia 'sex_MALE' che 'sex_FEMALE'
print(df.head())



# ======== Selezione delle caratteristiche per il clustering ========
features = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex_MALE"              #flag numerico per il sesso - da commentare per il secondo esperimento
]

X = df[features].values     #matrice delle feature selezionate che ha shape (n_pinguini, n_features)



# ======== Standardizzazione delle caratteristiche ========

#permette di avere tutte le feature sulla stessa scala -> altrimenti quelle con varianza più alta influenzano di più il clustering (peso maggiore)
scaler = StandardScaler()   #applica ad ongi colonna una formula di standardizzazione
X_scaled = scaler.fit_transform(X)


print("Shape X_scaled:", X_scaled.shape) #stampa la shape(n_pinguini x 5_features) del dataset standardizzato
print("Prima riga:", X_scaled[0])   #stampa la prima riga del dataset standardizzato, che corrisponde al primo pinguino



# ======== Applicazione di OPTICS ========
optics = OPTICS(
    min_samples=10,         #numero minimo di punti in un cluster per essere considerato denso
    xi=0.05,                #controlla la sensibilità della rilevazione dei cluster, più è basso più cluster vengono rilevati in quanto sensibile a variazioni piccole
    min_cluster_size=0.05   #ogni cluster deve avere almeno il 5% del numero totale di pinguini
)

optics.fit(X_scaled)


# Estrazione delle etichette dei cluster
labels = optics.labels_     #array delle etichette dei cluster assegnate da OPTICS a ogni pinguino: -1 indica rumore, 0,1,2,... indicano i vari cluster
unique, counts = np.unique(labels, return_counts=True) #calcola quante label esistono e quanti punguini appartengono a ciascuna label


# Stampa delle etichette uniche e la loro distribuzione
print("Label uniche:", unique)
print("Distribuzione:", dict(zip(unique, counts)))



# ======== Costruzione del reachability plot ========
ordering = optics.ordering_  #ordine di visita dei punti da parte di OPTICS, parte dal punto più denso, non è l'ordine originale dei pinguini nel dataset
reachability = optics.reachability_[ordering]   #reachability distance dei punti nell'ordine di visita


#Grafico del reachability plot
plt.figure(figsize=(10, 4))
plt.title("OPTICS - Reachability Plot (Penguins)")
plt.xlabel("Indice nell'ordine di visita")
plt.ylabel("Reachability distance")
plt.bar(np.arange(len(reachability)), reachability, width=1.0)
plt.tight_layout()
plt.show()



# ======== Creazione del grafico PCA per visualizzare i cluster ========
pca = PCA(n_components=2)   #riduzione a 2 dimensioni per la visualizzazione (da 5d a 2d)
X_pca = pca.fit_transform(X_scaled)
print("Varianza spiegata:", pca.explained_variance_ratio_)

plt.figure(figsize=(7, 6))
plt.title("Penguins - PCA 2D con cluster OPTICS")
plt.xlabel("PC1")
plt.ylabel("PC2")

for cluster_id in np.unique(labels):    #per ogni cluster
    mask = labels == cluster_id         #maschera booleana per selezionare i pinguini appartenenti al cluster corrente
    if cluster_id == -1:                #rumore
        lab = "Rumore (-1)"
        marker = "x"
        size = 40
        alpha = 0.9
    else:
        lab = f"Cluster {cluster_id}" #etichetta del cluster
        marker = "o"
        size = 20
        alpha = 0.7
    # Scatter plot dei punti appartenenti al cluster corrente
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=size,
        marker=marker,
        alpha=alpha,
        label=lab
    )

plt.legend()
plt.tight_layout()
plt.show()


df_clusters = df.copy()
df_clusters["cluster"] = labels


# ======== Risultati finali ========
print("\nDistribuzione dei pinguini per cluster:")
print(df_clusters["cluster"].value_counts().sort_index())

print("\nMedie per cluster:")
print(df_clusters.groupby("cluster")[features].mean())


print("\n fine del programma")
