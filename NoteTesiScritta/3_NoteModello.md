# Modello 3 resnet 1D
1. Configurazione iniziale
split_by_circuit: Determina se i dati devono essere suddivisi in base al circuito (ad esempio, utilizzando un circuito specifico come test set).
batch_size: Specifica la dimensione dei batch per l'addestramento.
num_epochs: Indica il numero massimo di epoche per l'addestramento.
patience: Stabilisce il numero di epoche consecutive senza miglioramenti dopo cui l'early stopping interrompe l'addestramento.

2. Caricamento e preprocessing dei dati
Caricamento dei dati: I dati di telemetria vengono caricati da file CSV. Se non vengono trovati file, il programma si interrompe.
Codifica di altre colonne categoriali: Anche le colonne track, driver e temp vengono codificate in valori numerici interi per essere utilizzate successivamente, ad esempio per raggruppare i dati o per suddividere il dataset.
Aggiunta di un indice temporale: Per ogni sessione (definita da track, driver e temp), viene aggiunta una colonna time_idx che rappresenta l'indice temporale progressivo. Questo permette di ordinare i dati in modo coerente.
Standardizzazione: Le feature numeriche vengono standardizzate per avere media 0 e deviazione standard 1. Questo aiuta il modello a convergere più rapidamente durante l'addestramento. Lo scaler utilizzato per questa operazione viene salvato su file per un eventuale riutilizzo.

3. Suddivisione del dataset
Suddivisione logica per circuito: Se configurato, i dati vengono suddivisi in base al circuito. Ad esempio, i dati di un circuito specifico (track == 2) vengono utilizzati come test set, mentre gli altri dati vengono suddivisi in training e validation set.
Suddivisione classica: In alternativa, i dati vengono suddivisi in proporzioni fisse (ad esempio, 60% per il training, 20% per la validation e 20% per il test).
Conversione in tensori: I dati vengono convertiti in tensori PyTorch per essere utilizzati come input per il modello. Vengono creati DataLoader per il training, la validation e il test, che gestiscono il caricamento dei dati in batch.

4. Definizione del modello ResNet 1D
Blocco residuo: Viene definito un blocco residuo che utilizza connessioni skip per migliorare la propagazione del gradiente. Ogni blocco include:
Due strati lineari con normalizzazione batch e attivazione ReLU.
Una connessione residua che somma l'input originale all'output del blocco.
Architettura del modello: Il modello è composto da:
Un livello di input che trasforma i dati iniziali in una rappresentazione di dimensione 128.
Una sequenza di blocchi residui per apprendere rappresentazioni complesse.
Un livello di output che riduce la rappresentazione a 64 unità e produce le probabilità di appartenenza alle classi.
Forward pass: Ogni input passa attraverso il livello di input, i blocchi residui e il livello di output per produrre la predizione finale.

5. Addestramento del modello
Configurazione del dispositivo: Il modello viene addestrato su GPU, se disponibile, altrimenti su CPU.
Funzione di perdita: Viene utilizzata la cross-entropy per calcolare la perdita tra le predizioni del modello e le etichette reali.
Ottimizzatore: L'ottimizzatore Adam viene utilizzato per aggiornare i pesi del modello durante l'addestramento.
Early Stopping: Una classe personalizzata monitora le prestazioni sul validation set e interrompe l'addestramento se non ci sono miglioramenti per un certo numero di epoche consecutive.


Accuratezza sul test (No divisione in base al circuito)
Validation Macro Precision: 0.8365
Epoch 26, Train Loss: 0.0734, Val Loss: 0.2181, Train Acc: 0.9706, Val Acc: 0.9191
Validation Macro Precision: 0.8426
Test Classification Report:
                      precision    recall  f1-score   support

           grip_loss       0.62      0.52      0.57       263
high_grip_accelerate       0.82      0.92      0.86      7157
 low_grip_accelerate       0.86      0.68      0.76      4297
             neutral       0.99      0.99      0.99     15283

            accuracy                           0.92     27000
           macro avg       0.82      0.78      0.80     27000
        weighted avg       0.92      0.92      0.92     27000



Con pesi modificati, per dare più peso alle classi con meno record

Validation Macro Precision: 0.8055
Test Classification Report:
                      precision    recall  f1-score   support

           grip_loss       0.52      0.56      0.54       263
high_grip_accelerate       0.86      0.80      0.83      7157
 low_grip_accelerate       0.72      0.82      0.76      4297
             neutral       0.99      0.98      0.99     15283

            accuracy                           0.90     27000
           macro avg       0.77      0.79      0.78     27000
        weighted avg       0.91      0.90      0.90     27000



Accuratezza sul test (Sì divisione in base al circuito)

Validation Macro Precision: 0.8447
Epoch 28, Train Loss: 0.0890, Val Loss: 0.1594, Train Acc: 0.9632, Val Acc: 0.9362
Grip_loss matrice di confusione 0.63
Test Classification Report:
                      precision    recall  f1-score   support

           grip_loss       0.85      0.63      0.72       709
high_grip_accelerate       0.90      0.92      0.91     13194
 low_grip_accelerate       0.81      0.82      0.81      4678
             neutral       0.99      0.98      0.98     26419

            accuracy                           0.94     45000
           macro avg       0.89      0.84      0.86     45000
        weighted avg       0.94      0.94      0.94     45000




Con pesi modificati, per dare più peso alle classi con meno record 
Molto buona matrice di confusione nella gripl_loss 0.70
Test Classification Report:
                      precision    recall  f1-score   support

           grip_loss       0.71      0.70      0.70       709
high_grip_accelerate       0.90      0.86      0.88     13194
 low_grip_accelerate       0.70      0.88      0.78      4678
             neutral       0.99      0.97      0.98     26419

            accuracy                           0.92     45000
           macro avg       0.83      0.85      0.84     45000
        weighted avg       0.93      0.92      0.92     45000

Epoch 27, Train Loss: 0.1984, Val Loss: 0.3166, Train Acc: 0.9481, Val Acc: 0.9207
Validation Macro Precision: 0.8170
