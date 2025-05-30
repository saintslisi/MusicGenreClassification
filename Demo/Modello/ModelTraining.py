from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import joblib
import torch.nn.functional as F
import torch.nn as nn
from Modello.preprocessing import *
# Modello di rete
class MusicGenreClassifier(nn.Module):
    def __init__(self):
        super(MusicGenreClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5944, 1000),
            nn.ReLU(),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 21)
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_features=5944, hidden_units=1024, out_classes=9, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.hidden_layer = nn.Linear(in_features, hidden_units)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_units, out_classes)
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        #x = self.dropout(x)
        return self.output_layer(x)