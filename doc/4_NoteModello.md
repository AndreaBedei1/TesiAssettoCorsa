# Modello 4 resnet 1D
Come il modello precedente ma prende gli input a sequenza cin finestra di 5


Accuratezza sul test (No divisione in base al circuito)
Classification Report:
                      precision    recall  f1-score   support

           grip_loss       0.79      0.54      0.64       420
high_grip_accelerate       0.86      0.91      0.89      7801
 low_grip_accelerate       0.88      0.78      0.83      3916
             neutral       0.97      0.98      0.98     14818

            accuracy                           0.92     26955
           macro avg       0.88      0.80      0.83     26955
        weighted avg       0.92      0.92      0.92     26955


Con pesi modificati, per dare più peso alle classi con meno record
Classification Report:
                      precision    recall  f1-score   support

           grip_loss       0.73      0.61      0.66       420
high_grip_accelerate       0.85      0.77      0.81      7801
 low_grip_accelerate       0.69      0.93      0.79      3916
             neutral       0.98      0.94      0.96     14818

            accuracy                           0.89     26955
           macro avg       0.81      0.81      0.81     26955
        weighted avg       0.90      0.89      0.89     26955


Accuratezza sul test (Sì divisione in base al circuito)

                      precision    recall  f1-score   support

           grip_loss       0.49      0.69      0.58       709
high_grip_accelerate       0.89      0.89      0.89     13189
 low_grip_accelerate       0.79      0.74      0.77      4676
             neutral       0.97      0.97      0.97     26351

            accuracy                           0.92     44925
           macro avg       0.79      0.82      0.80     44925
        weighted avg       0.92      0.92      0.92     44925



Con pesi modificati, per dare più peso alle classi con meno record 

Classification Report:
                      precision    recall  f1-score   support

           grip_loss       0.67      0.64      0.65       709
high_grip_accelerate       0.82      0.72      0.77     13189
 low_grip_accelerate       0.58      0.90      0.71      4676
             neutral       0.97      0.93      0.95     26351

            accuracy                           0.86     44925
           macro avg       0.76      0.80      0.77     44925
        weighted avg       0.88      0.86      0.86     44925