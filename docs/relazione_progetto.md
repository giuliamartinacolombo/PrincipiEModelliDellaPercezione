<p align="right">Giulia Martina Colombo</p>
<p align="right">Paolo Gavagni</p>
<p align="right">Laura Grosini</p>

# 🟢 Analisi delle Densità con OPTICS 🟢
## Esplorazione dei dataset e identificazione dei cluster attraverso l’algoritmo OPTICS
<hr style="height:5px; border:none; border-top:5px solid black;">

### 🗂️ Introduzione al Clustering

Il **clustering** è una tecnica di *machine learning non supervisionato* che mira a raggruppare dati simili tra loro, senza l’uso di etichette predefinite. In altre parole, l’algoritmo cerca di suddividere un insieme di punti in “gruppi” (cluster) tali che gli elementi all’interno di ciascun gruppo siano più simili tra loro rispetto a quelli appartenenti a gruppi differenti. Il **clustering** è ampiamente utilizzato in ambito commerciale, scientifico e tecnologico, ad esempio per segmentare clienti, raggruppare documenti simili, identificare pattern spaziali o analizzare immagini.

Il clustering offre diversi benefici chiave nell’analisi dei dati, tra cui:
* **Analisi esplorativa** --> aiuta a scoprire strutture nascoste nei dati, come segmenti di clienti, comunità o gruppi di comportamento.
* **Riduzione della complessità** --> raggruppare dati simili permette di sintetizzare l’informazione e facilitare ulteriori analisi.
* **Pre-elaborazione per altre tecniche** --> i cluster possono essere usati come feature in modelli supervisionati.
* **Applicazioni pratiche** --> nel trattamento di immagini (segmentazione), nell’elaborazione di documenti (raggruppamento per somiglianza) e in molti altri contesti.

Gli algoritmi di clustering si dividono principalmente in tre categorie:

* **Basati sulla distanza** (*es. K-Means, K-Medoids*) –-> raggruppano punti vicini nello spazio.
* **Basati sulla densità** (*es. DBSCAN, OPTICS*) –-> definiscono cluster come regioni di alta densità separate da zone di bassa densità.
* **Basati su modelli o gerarchie** (*es. Agglomerative Clustering, Gaussian Mixture*) –-> costruiscono cluster seguendo strutture gerarchiche o probabilistiche. 


### 🗂️ Perché utilizzare algoritmi basati sulla densità?

Mentre metodi come K-Means funzionano bene con cluster “sferici” e di dimensioni simili, molti dataset reali presentano cluster di forma irregolare e densità variabile.
Gli algoritmi basati sulla densità, come DBSCAN e OPTICS, superano queste limitazioni: identificano cluster di forma arbitraria e distinguono chiaramente tra punti rumorosi e cluster significativi.


### 🗂️ OPTICS: Clustering basato sulla densità

OPTICS (Ordering Points To Identify the Clustering Structure) è un algoritmo avanzato di clustering basato sulla densità, nato per superare i limiti di DBSCAN quando i cluster hanno densità diversa o forme complesse. A differenza di DBSCAN, non assegna subito etichette ai cluster, ma costruisce una rappresentazione ordinata dei punti chiamata reachability plot, che permette di identificare cluster e rumore in modo visivo e gerarchico.







