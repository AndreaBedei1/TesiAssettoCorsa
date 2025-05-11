# Modello 0
1. **Caricamento e preparazione dei dati**  
   - Lo script carica i dati di telemetria da uno o più file CSV che seguono un certo pattern di nome. Se non vengono trovati file, il programma segnala l’errore e interrompe l’esecuzione.  
   - I dati caricati subiscono un processo di “fix” Tolti 'wheel_slip_front_left', 'wheel_slip_front_right', 'wheel_slip_rear_left', 'wheel_slip_rear_right', 'current_time_str', 

3. **Codifica e trasformazioni preliminari**  
   - La colonna che identifica la variabile di output (chiamata “result”) viene trasformata da testo a valori numerici interi tramite un codificatore di etichette. In seguito, per l’apprendimento, questi valori numerici vengono ulteriormente convertiti in un formato one-hot.  
   - Anche le colonne che identificano il circuito, il pilota e la temperatura vengono convertite in valori numerici interi per un uso successivo negli split (train/test).  
   - Viene definita una serie di colonne di feature (un sottoinsieme di tutte le colonne) che conterranno i dati utili all’addestramento.

4. **Standardizzazione e salvataggio dello scaler**  
   - Si applica poi uno standardizzatore ai dati numerici di input per ottenere distribuzioni con media zero e deviazione standard unitaria, rendendo l’addestramento del modello più stabile.  
   - Lo scaler viene salvato su file per poter essere riutilizzato all’occorrenza.

5. **Suddivisione in train, validation e test (opzionalmente per circuito)**  
   - Lo script controlla se si desidera suddividere i dati in base a un circuito specifico. In caso affermativo, i record appartenenti al circuito scelto vengono usati come test set mentre gli altri costituiscono il gruppo train/validation.  
   - In alternativa, se non si è scelto lo split per circuito, i dati vengono divisi in proporzioni fisse (ad esempio 60% train, 20% validation e 20% test).  
   - Nel caso della divisione per circuito, il dataset viene “filtrato” in due parti: una parte da cui estrarre train e validation, e l’altra da usare come test. Poi train e validation vengono ricavati con un ulteriore meccanismo di slicing interno.

6. **Definizione e compilazione del modello**  
   - Viene creata una rete neurale sequenziale densa con diversi strati fully connected e operazioni di dropout per ridurre l’overfitting.  
   - L’architettura combina strati con svariate dimensioni (256, 128, 64, 32) e attivazioni ReLU, per catturare pattern non lineari nei dati.  
   - L’ultimo strato produce la probabilità di appartenenza alle quattro classi tramite l’attivazione softmax.  
   - Il modello viene compilato specificando l’ottimizzatore (Adam), la funzione di perdita (categoriale) e la metrica (accuratezza).
   -ELIMINZAIONE DI .drop(columns=["track", "driver", "temp"])

7. **Bilanciamento delle classi (class weights)**  
   - Prima dell’addestramento, si calcolano i pesi delle classi bilanciati in base alla loro frequenza. Ciò evita che, in presenza di classi poco rappresentate, il modello ne trascuri l’importanza.  
   - Questi pesi vengono convertiti in un dizionario che mappa l’indice di ciascuna classe al relativo fattore di ponderazione.

8. **Introduzione di early stopping**  
   - Si definisce un callback personalizzato che monitora le prestazioni sul validation set e interrompe l’addestramento quando non si osservano miglioramenti oltre un certo numero di epoche.  
   - Evita di far proseguire inutilmente il training se il modello si “satura” o inizia ad andare in overfitting.

9. **Fase di training**  
   - Viene avviato il processo di addestramento, specificando dati di input e output per il train, dimensione dei batch, numero di epoche e i pesi per il bilanciamento delle classi.  
   - Durante ogni epoca, il callback di early stopping valuta le prestazioni sul validation set e, se necessario, ferma l’addestramento prima dello script completo di epoche.


Accuratezza sul test (No divisione in base al circuito)
    Precision for label 'grip_loss': 0.57
    Precision for label 'high_grip_accelerate': 0.87
    Precision for label 'low_grip_accelerate': 0.67
    Precision for label 'neutral': 0.99
    Accuracy for label 'macro avg': 0.78
    Accuracy for label 'weighted avg': 0.90

    Epoch 17/50
    1265/1266 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9228 - loss: 0.2942
    Epoch 17: Mean Precision = 0.7380
    1266/1266 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - accuracy: 0.9228 - loss: 0.2942 - val_accuracy: 0.8601 - val_loss: 0.3297


Accuratezza sul test (Sì divisione in base al circuito)
    Precision for label 'grip_loss': 0.69
    Precision for label 'high_grip_accelerate': 0.84
    Precision for label 'low_grip_accelerate': 0.62
    Precision for label 'neutral': 1.00
    Accuracy for label 'macro avg': 0.78
    Accuracy for label 'weighted avg': 0.91

    Epoch 17/50
    1098/1125 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9171 - loss: 0.3166
    Epoch 17: Mean Precision = 0.7276 1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9171 - loss: 0.3165 - val_accuracy: 0.8835 - val_loss: 0.2617




Dipende dal contesto e dall'obiettivo del modello, ma in generale:
- **Accuracy**: È meglio che sia alta se il dataset è bilanciato e tutte le classi hanno la stessa importanza. L'accuracy misura la percentuale di predizioni corrette sul totale.
- **Precision per una classe specifica**: È meglio che sia alta se vuoi ridurre i falsi positivi per quella classe. Ad esempio, se la classe `grip_loss` è critica, una precision alta significa che il modello è affidabile quando predice `grip_loss`.
- **Recall per una classe specifica**: È meglio che sia alta se vuoi ridurre i falsi negativi per quella classe. Ad esempio, se è fondamentale identificare tutti i casi di `grip_loss`, una recall alta garantisce che il modello non perda esempi di quella classe.
- **Macro avg**: È meglio che sia alta se vuoi che il modello performi bene su tutte le classi, indipendentemente dalla loro frequenza.
- **Weighted avg**: È meglio che sia alta se vuoi che il modello performi bene considerando la distribuzione delle classi (dando più peso alle classi più frequenti).
In sintesi, la metrica più importante dipende dal tuo obiettivo specifico. Se alcune classi sono più critiche (ad esempio, `grip_loss`), potresti voler massimizzare precision e recall per quelle classi.


