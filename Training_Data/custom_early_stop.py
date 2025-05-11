from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import precision_score

class CustomEarlyStopping(Callback):
    def __init__(self, validation_data, patience=5):
        super().__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_val_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)

        # y_val può essere one-hot oppure interi
        if y_val.ndim == 2:
            y_val_true = np.argmax(y_val, axis=1)
        else:
            y_val_true = y_val

        precision_per_class = precision_score(
            y_val_true, y_val_pred, average=None, zero_division=0
        )
        mean_precision = np.mean(precision_per_class)

        print(f"Epoch {epoch + 1}: Mean Precision = {mean_precision:.4f}")

        if mean_precision > self.best_score:
            self.best_score = mean_precision
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Early stopping triggered. Restoring best weights.")
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

import torch

class CustomEarlyStoppingTorch:
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = -np.inf
        self.wait = 0
        self.best_model_state = None
        self.early_stop = False

    def evaluate(self, model, val_loader, device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        mean_precision = np.mean(precision_per_class)

        if self.verbose:
            print(f"\nValidation Macro Precision: {mean_precision:.4f}")

        return mean_precision

    def __call__(self, model, val_loader, device):
        score = self.evaluate(model, val_loader, device)

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait = 0
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print("Early stopping triggered. Restoring best weights.")
                self.early_stop = True
                model.load_state_dict(self.best_model_state)
