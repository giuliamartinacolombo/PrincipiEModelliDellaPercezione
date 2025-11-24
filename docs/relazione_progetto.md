<p align="right">Giulia Martina Colombo</p>
<p align="right">Paolo Gavagni</p>
<p align="right">Laura Grosini</p>

# 🟢 Analisi delle Densità con OPTICS 🟢
## Esplorazione dei dataset e identificazione dei cluster attraverso l’algoritmo OPTICS
---

### 🗂️ **Introduzione al Clustering**

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


### 🗂️ **Perché utilizzare algoritmi basati sulla densità?**

Mentre metodi come K-Means funzionano bene con cluster “sferici” e di dimensioni simili, molti dataset reali presentano cluster di forma irregolare e densità variabile.
Gli algoritmi basati sulla densità, come DBSCAN e OPTICS, superano queste limitazioni: identificano cluster di forma arbitraria e distinguono chiaramente tra punti rumorosi e cluster significativi.


### 🗂️ **OPTICS: Clustering basato sulla densità**

OPTICS (Ordering Points To Identify the Clustering Structure) è un algoritmo avanzato di clustering basato sulla densità, nato per superare i limiti di DBSCAN quando i cluster hanno densità diversa o forme complesse. A differenza di DBSCAN, non assegna subito etichette ai cluster, ma costruisce una rappresentazione ordinata dei punti chiamata reachability plot, che permette di identificare cluster e rumore in modo visivo e gerarchico.

**CONCETTI CHIAVE**

#### **Core Distance**

La *core distance* rappresenta una misura fondamentale per capire se un punto può essere considerato parte del “cuore” di un cluster.
In pratica, serve a valutare quanto un punto sia immerso in una zona densa: più vicini ha attorno a sé, più è probabile che appartenga a un cluster ben formato.

Dal punto di vista matematico, per un punto ( p ), la core distance è la distanza che lo separa dal suo **MinPts-esimo vicino più vicino**. Questo valore riflette il livello di densità locale:

* se il punto ha almeno *MinPts* vicini in un raggio ragionevole, è abbastanza “circondato” da altri punti e diventa un **core point**, cioè un punto in grado di espandere un cluster;
* al contrario, se i vicini sono pochi o troppo distanti, il punto non ha sufficiente densità attorno a sé e non può dare origine a un cluster.

Un modo intuitivo per visualizzarlo: immagina un punto circondato da almeno 5 altri punti molto vicini → è in una zona densa, quindi è un core point. Se invece è quasi isolato, non può contribuire alla creazione di un cluster compatto.

#### **Reachability Distance**

La *reachability distance* è un modo per quantificare **quanto è “raggiungibile” un punto a partire da un altro punto che si trova in una zona densa**.
È una misura più flessibile della semplice distanza geometrica, perché tiene conto del livello di densità del punto di partenza.

Si calcola prendendo il massimo tra:

* la core distance del punto di partenza (che descrive quanto è denso l’ambiente locale),
* la distanza effettiva tra i due punti.

In formula:

<p align="center"><strong><em>reachability(p, q)=\max(core_distance(p),distance(p,q))</em></strong></p>

Cosa significa in pratica?
Se ci muoviamo all’interno di un cluster ben definito, i punti saranno tutti relativamente vicini e circondati da altri punti → la reachability distance rimane bassa.
Man mano che ci spostiamo verso il bordo del cluster, o verso zone più vuote, le distanze aumentano → la reachability cresce e segnala un cambio di densità.

È proprio questa variazione che permette a OPTICS di distinguere zone dense (cluster) da punti isolati o rumore.

#### **Reachability Plot**

Il *reachability plot* è uno degli elementi più caratteristici e potenti di OPTICS.
Si tratta di un grafico dove i punti non vengono semplicemente mostrati nello spazio originale, ma **ordinati secondo la sequenza con cui l’algoritmo li visita**, e rappresentati in base alla loro reachability distance.

Visualmente, il grafico funziona così:

* **Le “vallate”** indicano regioni a bassa reachability distance, quindi zone dense → corrispondono ai cluster.
* **I “picchi”** rappresentano punti con alta reachability distance → spesso rumore o transizioni tra cluster.

Questo tipo di rappresentazione è estremamente utile perché permette di osservare:

* cluster di forma qualsiasi,
* cluster con densità molto diverse tra loro,
* cluster annidati uno dentro l’altro,
* e la distribuzione del rumore nel dataset.

È un livello di dettaglio che DBSCAN non può offrire, proprio perché OPTICS non si limita a “tagliare” i cluster con un singolo valore di eps, ma lascia emergere la loro struttura direttamente dal grafico.




