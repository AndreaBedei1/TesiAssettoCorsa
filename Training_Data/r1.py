import torch.nn as nn
import torch.nn.functional as F

# Strato di Input:
# Uno strato lineare che proietta i dati di input in uno spazio di 128 dimensioni.
# Seguito da una funzione di attivazione ReLU e una normalizzazione batch.

# Blocchi Residui (Residual Blocks):
# Una sequenza di blocchi residui (il numero è determinato dal parametro depth).
# Ogni blocco residuo contiene:
# Due strati lineari con dimensione costante (128).
# Una funzione di attivazione ReLU dopo il primo strato.
# Normalizzazione batch dopo ogni strato lineare.
# Una connessione residua che somma l'input originale al risultato del blocco.

# Strato di Output:
# Uno strato lineare che riduce la dimensione da 128 a 64.
# Seguito da una funzione di attivazione ReLU.
# Uno strato lineare finale che mappa i 64 nodi ai nodi di output corrispondenti al numero di classi.

# Forward Pass:
# I dati passano attraverso lo strato di input, i blocchi residui e infine lo strato di output per produrre la predizione finale

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

class ResNet1DTabular(nn.Module):
    def __init__(self, input_dim, num_classes, depth=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(depth)])
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)