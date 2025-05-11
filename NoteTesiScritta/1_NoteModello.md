# Modello 1 LSTM

1. Preprocessing dei dati
Caricamento dei dati: I dati di telemetria vengono caricati da file CSV utilizzando un pattern specifico per individuare i file. Ogni file contiene informazioni relative a un pilota, un circuito e una temperatura specifica. Questi metadati vengono estratti dal nome del file e aggiunti come colonne al dataset.
Conversione del tempo: La colonna che rappresenta il tempo in formato stringa (minuti:secondi:millisecondi) viene convertita in millisecondi per facilitarne l'uso nei calcoli.
Pulizia dei dati: Vengono rimossi i record con valori di velocità negativi, poiché non sono validi. Eliminazione colonne dello slip.
Raggruppamento dei dati: I dati vengono raggruppati in base a combinazioni di pilota, circuito e temperatura, creando così sessioni di guida specifiche.
2. Creazione delle sequenze temporali
Aggiunta di un indice temporale: Per ogni sessione, viene aggiunta una colonna time_idx che rappresenta l'indice temporale progressivo. Questo permette di ordinare i dati in modo coerente e di creare sequenze temporali.
Creazione delle finestre temporali: Per ogni sessione, vengono create finestre temporali di lunghezza predefinita (window_size). Ogni finestra rappresenta una sequenza di dati consecutivi:
Le feature numeriche di ciascuna finestra vengono salvate come input.
La classe associata all'ultimo elemento della finestra viene salvata come target.
3. Suddivisione del dataset
Suddivisione logica per circuito: Se configurato, i dati vengono suddivisi in base al circuito. Ad esempio, i dati di un circuito specifico possono essere utilizzati come test set, mentre gli altri dati vengono suddivisi in training e validation set.
Suddivisione classica: In alternativa, i dati vengono suddivisi in proporzioni fisse (ad esempio, 60% per il training, 20% per la validation e 20% per il test).
4. Modelli di machine learning
Modello LSTM (TensorFlow/Keras)
Architettura: Viene definito un modello sequenziale con i seguenti strati:
Un livello LSTM per apprendere le dipendenze temporali all'interno delle sequenze.
Strati di dropout per ridurre l'overfitting.
Strati densi con attivazione ReLU per apprendere rappresentazioni complesse.
Un livello di output con attivazione softmax per produrre le probabilità di appartenenza alle classi.
Compilazione: Il modello viene compilato con l'ottimizzatore Adam e la funzione di perdita categoriale.
Modello ResNet 1D (PyTorch)
Architettura: Viene definito un modello ResNet per dati tabulari/sequenziali:
Un livello di input che trasforma i dati iniziali.
Blocchi residui che utilizzano connessioni skip per migliorare la propagazione del gradiente.
Un livello di output che produce le probabilità di appartenenza alle classi.
Forward pass: Ogni input passa attraverso i blocchi residui e viene trasformato in una rappresentazione finale per la classificazione.
5. Early Stopping personalizzato
TensorFlow/Keras
Callback personalizzato: Monitora la precisione media sulle classi nel validation set. Se non ci sono miglioramenti per un certo numero di epoche consecutive, l'addestramento viene interrotto e i pesi migliori vengono ripristinati.
Logica:
Alla fine di ogni epoca, vengono calcolate le predizioni sul validation set.
La precisione media viene confrontata con il miglior valore precedente.
Se non ci sono miglioramenti, viene incrementato un contatore (wait). Quando il contatore supera una soglia (patience), l'addestramento si interrompe.

Logica:
Durante la validazione, vengono calcolate le predizioni e la precisione media.
Se la precisione migliora, lo stato del modello viene salvato.
Se non ci sono miglioramenti per un certo numero di epoche consecutive, l'addestramento si interrompe e lo stato migliore viene ripristinato.
6. Addestramento
TensorFlow/Keras: Il modello LSTM viene addestrato utilizzando i dati di training e validation. Durante l'addestramento, il callback di early stopping monitora le prestazioni e può fermare il processo se necessario.
PyTorch: Il modello ResNet viene addestrato utilizzando un ciclo di training e validazione. La classe di early stopping monitora le prestazioni e può interrompere l'addestramento.
7. Salvataggio del modello
I modelli addestrati vengono salvati su file per poter essere riutilizzati in futuro senza doverli riaddestrare.
8. Valutazione finale
I modelli vengono valutati sul test set per calcolare le prestazioni complessive. Vengono generate le previsioni per il test set, che saranno utilizzate per ulteriori analisi (ad esempio, report di classificazione e matrice di confusione).

Accuratezza sul test (No divisione in base al circuito)
Epoch 23/50
1263/1264 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9612 - loss: 0.0998Epoch 23: Mean Precision = 0.8616
1264/1264 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - accuracy: 0.9612 - loss: 0.0998 - val_accuracy: 0.9587 - val_loss: 0.1117

Test Accuracy: 0.93
843/843 ━━━━━━━━━━━━━━━━━━━━ 1s 945us/step
Precision for label 'grip_loss': 0.83
Precision for label 'high_grip_accelerate': 0.88
Precision for label 'low_grip_accelerate': 0.87
Precision for label 'neutral': 0.98
Accuracy for label 'macro avg': 0.89
Accuracy for label 'weighted avg': 0.93 

Con pesi modificati, per dare più peso alle classi con meno record (foto)
Precision for label 'grip_loss': 0.62
Precision for label 'high_grip_accelerate': 0.86
Precision for label 'low_grip_accelerate': 0.75
Precision for label 'neutral': 0.99
Accuracy for label 'macro avg': 0.80
Accuracy for label 'weighted avg': 0.91



Accuratezza sul test (Sì divisione in base al circuito)

Molto buona anche matrice di confusione

Precision for label 'grip_loss': 0.85
Precision for label 'high_grip_accelerate': 0.91
Precision for label 'low_grip_accelerate': 0.82
Precision for label 'neutral': 0.98
Accuracy for label 'macro avg': 0.89
Accuracy for label 'weighted avg': 0.94

Con pesi modificati, per dare più peso alle classi con meno record (foto) .76 girp_loss nella matrice di confusione
1114/1124 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9403 - loss: 0.2150
Epoch 20: Mean Precision = 0.6663
1124/1124 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - accuracy: 0.9403 - loss: 0.2150 - val_accuracy: 0.8861 - val_loss: 0.3665

Precision for label 'grip_loss': 0.54
Precision for label 'high_grip_accelerate': 0.88
Precision for label 'low_grip_accelerate': 0.71
Precision for label 'neutral': 0.99
Accuracy for label 'macro avg': 0.78
Accuracy for label 'weighted avg': 0.92