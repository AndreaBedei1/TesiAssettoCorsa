# Modello 2 Transformes

Questo codice implementa un modello Transformer per classificare sequenze temporali basate su dati di telemetria automobilistica. Di seguito è descritta in dettaglio la logica delle parti principali:

---

### **1. Configurazione iniziale**
Il codice inizia con la configurazione di alcuni parametri fondamentali:
- **`split_by_circuit`**: Determina se i dati devono essere suddivisi in base al circuito (ad esempio, utilizzando un circuito specifico come test set).
- **`window_size`**: Specifica la lunghezza delle finestre temporali che verranno create per rappresentare le sequenze.

---

### **2. Caricamento e preprocessing dei dati**
- **Caricamento dei dati**: I dati di telemetria vengono caricati da file CSV utilizzando un pattern specifico. Ogni file contiene informazioni relative a un pilota, un circuito e una temperatura specifica, che vengono estratte dal nome del file e aggiunte come colonne al dataset.
- **Pulizia dei dati**: Una funzione dedicata esegue operazioni di preprocessing sui dati, come la conversione del tempo in millisecondi, la rimozione di valori di velocità negativi e l'eliminazione di colonne non rilevanti.
- **Codifica delle etichette**: La colonna `result`, che rappresenta la variabile target (ad esempio, lo stato del grip), viene trasformata in valori numerici interi tramite un codificatore di etichette.
- **Codifica di altre colonne categoriali**: Anche le colonne `track`, `driver` e `temp` vengono codificate in valori numerici interi per essere utilizzate successivamente, ad esempio per raggruppare i dati o per suddividere il dataset.
- **Standardizzazione**: Le feature numeriche vengono standardizzate per avere media 0 e deviazione standard 1. Questo aiuta il modello a convergere più rapidamente durante l'addestramento. Lo scaler utilizzato per questa operazione viene salvato su file per un eventuale riutilizzo.

---

### **3. Creazione delle sequenze temporali**
- **Aggiunta di un indice temporale**: Per ogni sessione (definita da `track`, `driver` e `temp`), viene aggiunta una colonna `time_idx` che rappresenta l'indice temporale progressivo. Questo permette di ordinare i dati in modo coerente e di creare sequenze temporali.
- **Raggruppamento dei dati**: I dati vengono raggruppati in base a `track`, `driver` e `temp`, creando così sessioni di guida specifiche.
- **Creazione delle finestre temporali**: Per ogni gruppo, vengono create finestre temporali di lunghezza `window_size`. Ogni finestra rappresenta una sequenza di dati consecutivi:
  - Le feature numeriche di ciascuna finestra vengono salvate come input.
  - La classe associata all'ultimo elemento della finestra viene salvata come target.

---

### **4. Suddivisione del dataset**
- **Suddivisione logica per circuito**: Se configurato, i dati vengono suddivisi in base al circuito. Ad esempio, i dati di un circuito specifico possono essere utilizzati come test set, mentre gli altri dati vengono suddivisi in training e validation set.
- **Suddivisione classica**: In alternativa, i dati vengono suddivisi in proporzioni fisse (ad esempio, 60% per il training, 20% per la validation e 20% per il test).
- **Conversione in array**: Le sequenze e le etichette vengono convertite in array NumPy per essere utilizzate come input per il modello.

---

### **5. Definizione del modello Transformer**
- **Blocco Transformer Encoder**: Viene definito un blocco encoder Transformer che include:
  - **Multi-Head Attention**: Permette al modello di catturare relazioni a lungo raggio tra gli elementi della sequenza.
  - **Residual Connection e Layer Normalization**: Stabilizzano l'addestramento e preservano le informazioni originali.
  - **Feed-Forward Network (FFN)**: Una rete convoluzionale 1D con due livelli per trasformare le rappresentazioni.
- **Architettura del modello**: Il modello è composto da:
  - Un livello di input che accetta sequenze temporali.
  - Due blocchi Transformer Encoder per catturare le dipendenze temporali.
  - Un livello di pooling globale per ridurre la dimensione temporale.
  - Strati densi con attivazione ReLU per apprendere rappresentazioni complesse.
  - Un livello di output con attivazione softmax per produrre le probabilità di appartenenza alle classi.
- **Compilazione**: Il modello viene compilato specificando l'ottimizzatore Adam, la funzione di perdita categoriale e la metrica di accuratezza.

---

### **6. Addestramento del modello**
- **Early Stopping**: Viene utilizzato un callback personalizzato per interrompere l'addestramento se le prestazioni sul validation set non migliorano per un certo numero di epoche consecutive.
- **Training**: Il modello viene addestrato utilizzando i dati di training e validation. Durante l'addestramento, il callback monitora le prestazioni e può fermare il processo se necessario.

---

### **7. Salvataggio del modello**
- Una volta completato l'addestramento, il modello viene salvato su file per poter essere riutilizzato in futuro senza doverlo riaddestrare.

---

### **8. Valutazione finale**
- Il modello viene valutato sul test set per calcolare le prestazioni complessive. Inoltre, vengono generate le previsioni per il test set, che saranno utilizzate per ulteriori analisi (ad esempio, report di classificazione e matrice di confusione).


Accuratezza sul test (No divisione in base al circuito)
Molto bassa matrice nella diagonale di grip_loss

Epoch 28/50
1261/1264 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.9323 - loss: 0.1758Epoch 28: Mean Precision = 0.8347
1264/1264 ━━━━━━━━━━━━━━━━━━━━ 29s 23ms/step - accuracy: 0.9323 - loss: 0.1758 - val_accuracy: 0.9403 - val_loss: 0.1617

Test Accuracy: 0.91
843/843 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step
Precision for label 'grip_loss': 0.81
Precision for label 'high_grip_accelerate': 0.85
Precision for label 'low_grip_accelerate': 0.87
Precision for label 'neutral': 0.94
Precision for label 'macro avg': 0.87
Precision for label 'weighted avg': 0.90

Con pesi modificati, per dare più peso alle classi con meno record
Matrice confusione non buona
Precision for label 'grip_loss': 0.50
Precision for label 'high_grip_accelerate': 0.75
Precision for label 'low_grip_accelerate': 0.69
Precision for label 'neutral': 0.96
Precision for label 'macro avg': 0.72
Precision for label 'weighted avg': 0.85



Accuratezza sul test (Sì divisione in base al circuito)

Matrice di confusione 0.52 per la loss_grip
Precision for label 'grip_loss': 0.82
Precision for label 'high_grip_accelerate': 0.89
Precision for label 'low_grip_accelerate': 0.78
Precision for label 'neutral': 0.95
Precision for label 'macro avg': 0.86
Precision for label 'weighted avg': 0.91




Con pesi modificati, per dare più peso alle classi con meno record 
Matrice di confusione 0.78 la grip_loss

Test Accuracy: 0.91
1404/1404 ━━━━━━━━━━━━━━━━━━━━ 15s 11ms/step
Precision for label 'grip_loss': 0.58
Precision for label 'high_grip_accelerate': 0.87
Precision for label 'low_grip_accelerate': 0.72
Precision for label 'neutral': 0.98
Precision for label 'macro avg': 0.79
Precision for label 'weighted avg': 0.91