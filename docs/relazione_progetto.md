<p align="right">Giulia Martina Colombo</p>
<p align="right">Paolo Gavagni</p>
<p align="right">Laura Grosini</p>

# 🟢 Analisi delle Densità con OPTICS 🟢
## Esplorazione dei dataset e identificazione dei cluster attraverso l’algoritmo OPTICS
---

### 🗂️ **Introduzione al Clustering**

Il **clustering** è una tecnica di *machine learning non supervisionato* che mira a raggruppare dati simili tra loro, senza l’uso di etichette predefinite. In altre parole, l’algoritmo cerca di suddividere un insieme di punti in “gruppi” (cluster) tali che gli elementi all’interno di ciascun gruppo siano più simili tra loro rispetto a quelli appartenenti a gruppi differenti. Il **clustering** è ampiamente utilizzato in ambito commerciale, scientifico e tecnologico, ad esempio per segmentare clienti, raggruppare documenti simili, identificare pattern spaziali o analizzare immagini.

<span><em>[Nel machine learning, esistono due grandi categorie di algoritmi:
* Apprendimento supervisionato,
* Apprendimento non supervisionato.
Nell’apprendimento supervisionato, l’algoritmo impara da un insieme di dati etichettati.
Significa che per ogni esempio fornito, conosci già la “risposta corretta”.
Per esempio, se vengono mostrate tante immagini di gatti e cani, ognuna con la sua etichetta (“gatto”, “cane”), l’algoritmo impara a riconoscere da solo la differenza e poi può classificare nuove immagini.
👉 Qui l’obiettivo è predire un’etichetta nota su nuovi dati.
 
Nel clustering, invece, non abbiamo etichette.
L’algoritmo riceve solo i dati grezzi — punti, immagini, segnali, ecc. — e deve scoprire da solo se esistono gruppi o strutture nascoste.
Ad esempio, vengono dati mille punti distribuiti nello spazio, senza dire quali “appartengono insieme”; l’algoritmo di clustering cerca di capire da solo quali punti sono simili o vicini e li raggruppa.
👉 Qui l’obiettivo non è “predire”, ma capire la struttura interna dei dati, trovare pattern o regolarità.]
</em></span>

Il clustering offre diversi benefici chiave nell’analisi dei dati, tra cui:
* **Analisi esplorativa** --> aiuta a scoprire strutture nascoste nei dati, come segmenti di clienti, comunità o gruppi di comportamento.
* **Riduzione della complessità** --> raggruppare dati simili permette di sintetizzare l’informazione e facilitare ulteriori analisi.
* **Pre-elaborazione per altre tecniche** --> i cluster possono essere usati come feature in modelli supervisionati.
* **Applicazioni pratiche** --> nel trattamento di immagini (segmentazione), nell’elaborazione di documenti (raggruppamento per somiglianza) e in molti altri contesti.

Gli algoritmi di clustering si dividono principalmente in tre categorie:

* **Basati sulla distanza** (*es. K-Means, K-Medoids*) –-> raggruppano punti vicini nello spazio.
* **Basati sulla densità** (*es. DBSCAN, OPTICS*) –-> definiscono cluster come regioni di alta densità separate da zone di bassa densità.
* **Basati su modelli o gerarchie** (*es. Agglomerative Clustering, Gaussian Mixture*) –-> costruiscono cluster seguendo strutture gerarchiche o probabilistiche. 

---

### 🗂️ **Perché utilizzare algoritmi basati sulla densità?**

Mentre metodi come K-Means funzionano bene con cluster “sferici” e di dimensioni simili, molti dataset reali presentano cluster di forma irregolare e densità variabile.
Gli algoritmi basati sulla densità, come DBSCAN e OPTICS, superano queste limitazioni: identificano cluster di forma arbitraria e distinguono chiaramente tra punti rumorosi e cluster significativi.

---

### 🗂️ **OPTICS: Clustering basato sulla densità**

OPTICS (Ordering Points To Identify the Clustering Structure) è un algoritmo avanzato di clustering basato sulla densità, nato per superare i limiti di DBSCAN quando i cluster hanno densità diversa o forme complesse. A differenza di DBSCAN, non assegna subito etichette ai cluster, ma costruisce una rappresentazione ordinata dei punti chiamata reachability plot, che permette di identificare cluster e rumore in modo visivo e gerarchico.

Si tratta di un algoritmo di fatto considerato superiore ad algoritmi come K-Means e il sopra citato DBSCAN per due motivi di rilievo:
* Non necessita, contrariamente a K-Means, di definire a priori il numero di cluster, in quanto li identifica automaticamente in ordine di densità.
* Lavora su densità variabili di dati, contrariamente a K-Means che assume a priori che i cluster abbiano densità e forme simili, e DBSCAN che tende a fare fatica se i cluster hanno densità variabili.

**CONCETTI CHIAVE**

#### **Core Points**

I *core points* sono quei data point che costituiscono la base su cui si costruiscono i cluster. Hanno un numero sufficiente di punti vicini  (almeno MinPts in uno specifico raggio), che costituiscono dunque una regione densa. Sono quindi fondamentali per definire dove un cluster ha inizio e come si estende in modo continuo.

#### **Border Points**

I *border points* si trovano ai margni di un cluster: non hanno abbastanza vicini per essere considerabili core points, ma rientrano nel "vicinato" di uno o più core points. Per questo vengono assegnati ai cluster, pur trovandosi nelle zone meno dense, e contribuiscono a delinearne il bordo.

#### **Noise Points**

I cosiddetti *noise points* stanno invece fuori dai cluster in quanto non hanno abbastanza vicini per essere definiti core points, e non rientrano nel "vicinato" di nessuno di essi. Per questo sono considerati **outlier/anomalie** e spesso corrispondono a dati irregolari o molto dispersi che non si inseriscono in nessun cluster. Riconoscerli è importante per valutare la qualità dei dati e filtrare le informazioni irrilevanti o inaccurate.

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

*Cosa significa in pratica?
Se ci muoviamo all’interno di un cluster ben definito, i punti saranno tutti relativamente vicini e circondati da altri punti → la reachability distance rimane bassa.
Man mano che ci spostiamo verso il bordo del cluster, o verso zone più vuote, le distanze aumentano → la reachability cresce e segnala un cambio di densità.*

È proprio questa variazione che permette a OPTICS di distinguere zone dense (cluster) da punti isolati o rumore.

#### **Reachability Plot**

Il *reachability plot* è uno degli elementi più caratteristici e potenti di OPTICS, in grado di elevarlo rispetto agli altri algoritmi dello stesso tipo.
Si tratta di un grafico dove i punti non vengono semplicemente mostrati nello spazio originale, ma **ordinati secondo la sequenza con cui l’algoritmo li visita**, e rappresentati in base alla loro reachability distance.
Visualmente, il grafico funziona così:

* **Le “vallate”** (o local minima) indicano regioni a bassa reachability distance, quindi zone dense → corrispondono ai cluster. I drop significativi segnano l'ingresso in una zona densa e i minimi locali ne rappresentano il "cuore".
* **I “picchi”** (o local maxima) rappresentano punti con alta reachability distance → spesso indicano rumore, outlier, o transizioni tra cluster, cioè i confini tra una regione densa e l'altra.

Questo tipo di rappresentazione è estremamente utile perché permette di osservare:

* cluster di forma qualsiasi,
* cluster con densità molto diverse tra loro,
* cluster annidati uno dentro l’altro,
* e la distribuzione del rumore nel dataset.

È un livello di dettaglio che DBSCAN non può offrire, proprio perché OPTICS non si limita a “tagliare” i cluster con un singolo valore di eps, ma lascia emergere la loro struttura direttamente dal grafico.

---

### 🗂️ **Funzionamento dell’algoritmo**

Il processo con cui OPTICS analizza un dataset può essere immaginato come una sorta di *esplorazione guidata* dello spazio dei punti, dove l’algoritmo visita ogni punto seguendo un ordine che riflette la densità dell’area in cui si trova. Questo permette di ottenere una visione molto accurata della struttura dei cluster.

#### **1. SELEZIONE DEL PUNTO INIZIALE**

L’algoritmo comincia scegliendo un punto qualsiasi che non sia ancora stato visitato. Una volta selezionato, calcola quanti altri punti si trovano entro un certo raggio massimo, chiamato **Eps**.
Questi punti vicini costituiranno la base per valutare quanto è densa la regione attorno al punto.


#### **2. VERIFICA DELLA DENSITA'**

A questo punto OPTICS controlla quanti vicini ha il punto selezionato:

* se il numero di punti vicini è **almeno MinPts**, allora siamo in una zona densa, e il punto viene classificato come **core point**;
* se invece i vicini sono troppo pochi, il punto non è abbastanza immerso nella densità e quindi viene considerato **non-core**.

È importante notare che un punto non-core può comunque far parte di un cluster, ma **non è in grado di espandere un cluster da solo**.

#### **3. CALCOLO DELLA REACHABILITY DISTANCE**

Se il punto è un core point, OPTICS procede a valutare la “raggiungibilità” dei suoi vicini.
Per ogni vicino non ancora visitato si calcola la **reachability distance**, che indica quanto è facile raggiungerlo dal punto corrente.

Più la reachability distance è bassa, più quel vicino si trova in un'area densa e quindi più è probabile che appartenga a un cluster.

Tutti questi vicini vengono inseriti in una struttura dati chiamata **priority queue**, che li ordina automaticamente dal più “raggiungibile” al meno raggiungibile.
In questo modo OPTICS ha sempre a disposizione il prossimo punto più naturale da visitare.

#### **4. ORDINE DEI PUNTI**

L’algoritmo continua quindi prelevando dalla coda il punto con la reachability distance più bassa e lo elabora.
Questo processo si ripete fino a quando non sono stati visitati tutti i punti del dataset.

La sequenza di visita generata in questo modo è fondamentale: costituisce infatti l’**ordine di raggiungibilità**, ovvero la base per costruire il reachability plot.

#### **5. IDENTIFICAZIONE DI CLUSTER E RUMORE**

Terminata l’analisi, OPTICS rappresenta graficamente la reachability distance dei punti secondo l’ordine in cui sono stati visitati.

Nel **reachability plot**:

* le **vallate** indicano regioni di bassa distanza di raggiungibilità, quindi aree dense → *cluster*;
* i **picchi** rappresentano improvvisi aumenti della distanza, tipici delle zone poco dense → *rumore o punti isolati*.

L'individuazione dei **border points** permette inoltre di modellare una transizione naturale fra diversi gruppi di dati, per fare sì che i cluster non siano separati artificialmente.

Ciò che rende OPTICS così potente è che questa rappresentazione permette di “leggere” la struttura dei cluster **a densità variabile**, e di individuare cluster a diversi livelli di dettaglio **senza dover fissare un valore unico di epsilon**, come avviene invece in DBSCAN. 

---

### 🗂️ **Confronto tra DBSCAN e OPTICS**

Per comprendere appieno le potenzialità di OPTICS, è utile metterlo a confronto con l’algoritmo da cui deriva: DBSCAN. Sebbene entrambi appartengano alla famiglia dei metodi basati sulla densità, differiscono per capacità, flessibilità e tipo di risultati prodotti. La tabella seguente mette in evidenza le principali differenze, evidenziando i punti di forza e i limiti di ciascun algoritmo.

**Caratteristica** --> Gestione densità variabili  
**DBSCAN** --> Richiede epsilon unico  
**OPTICS** --> Cluster di densità diversa identificabili    

**Caratteristica** --> Identificazione cluster  
**DBSCAN** --> Assegna cluster direttamente senza gerarchia  
**OPTICS** --> Usa reachability plot, supporta struttura gerarchica  

**Caratteristica** --> Struttura gerarchica  
**DBSCAN** --> Non supportata  
**OPTICS** --> Supporta cluster annidati    

**Caratteristica** --> Complessità computazionale    
**DBSCAN** --> Minore  
**OPTICS** --> Più alta per ordinamento e calcolo reachability    

**Caratteristica** --> Uso memoria  
**DBSCAN** --> Minore  
**OPTICS** --> Più elevato (mantiene una coda prioritaria)  

**Caratteristica** --> Parametri  
**DBSCAN** --> Richiede tuning accurato di epsilon e MinPts  
**OPTICS** --> Ridotta sensibilità a epsilon  

**Caratteristica** --> Rumore    
**DBSCAN** --> Identificato direttamente    
**OPTICS** --> Rappresentato dai picchi nel reachability plot 

**Caratteristica** --> Scalabilità    
**DBSCAN** --> Moderata, potrebbe fare fatica con dati di molte dimensioni   
**OPTICS** --> Poco scalabile su dataset ampi in quanto complesso dal punto di vista computazionale per via dei molteplici calcoli di distanze

---

### 🗂️ **Applicazioni pratiche**

OPTICS è particolarmente utile in scenari dove i cluster hanno densità differente o forme complesse:

* **Segmentazione clienti** --> raggruppamento di clienti in base a comportamento, preferenze, aspetti demografici nei contesti di e-commerce e retail. In questo modo è possibile nel pratico creare delle raccomandazioni su misura per ogni cliente o gruppo.
* **Individuazione di anomalie in sistemi di rilevazione di frodi** --> Per sistemi che devono identificare transazioni fraudolente è utile perchè riesce ad evidenziare pettern sospetti che si discostano dal comportamento "normale". In un contesto bancario, ad esempio, può rivelare transazioni anomale individuando cluster inusuali o punti isolati sulla base di informazioni come importo, localizzazione e momento dell'operazione.
* **Dati geospaziali** --> la sua nota flessibilità permette un'analisi più accurata delle relazioni spaziali in ambiti come urban planning, scelta delle location per il retail e studi ambientali. Ad esempio, è molto utile nel *real estate* per analizzare la richiesta di alloggi in diverse zone, oppure per agenzie ambientali che vogliono individuare cluster basati sui livelli di inquinamento in un'area geografica.
* **Analisi documenti** --> raggruppamento di testi simili.
* **Elaborazione immagini** --> identificazione di regioni di interesse o segmentazione oggetti.

---

### 🗂️ **Conclusioni**

L’algoritmo OPTICS rappresenta uno strumento potente per l’analisi dei dati complessi grazie alla sua capacità di:

* Gestire cluster con densità variabili e di forma arbitraria, fornendo flessibilità ben oltre ad algoritmi come K-Means.
* Offrire una rappresentazione gerarchica e flessibile dei cluster tramite reachability plot, fornendo un supporto in termini di interpretabilità e decisione.
* Supportare applicazioni multidisciplinari, dall’analisi commerciale alla segmentazione di immagini.
* Individuare il rumore e gli outliers distinguendo regioni sparse da cluster densi, migliorando di fatto la rilevanza dei pattern.

Ha però anche delle limitazioni, che vanno tenute in considerazione quando si decide se utilizzarlo:

* È computazionalmente pesante sui dataset molto grandi, perchè richiede molti calcoli di distanza ripetuti; questo può limitarne la scalabilità e l'uso in scenari real-time.
* È sensibile alla scelta dei parametri e della metrica di distanza: selezionare valori e misure adeguate è cruciale per ottenere risultati significativi.
* Le prestazioni peggiorano con dati ad alta dimensionalità, utili per migliorare sia l'efficienza sia l'acuratezza dell'analisi.

Nonostante una maggiore complessità computazionale rispetto a DBSCAN, OPTICS fornisce una visione più dettagliata della struttura dei dati, rendendolo ideale per dataset complessi e ricchi di pattern nascosti.

