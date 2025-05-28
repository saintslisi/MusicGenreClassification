# Importa la libreria del progetto (definisce funzioni e classi personalizzate)
import libProject 
# Importa PyTorch e librerie correlate
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
# Importa metriche di valutazione da scikit-learn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import time
import pandas as pd
import ssl
import json
import matplotlib.pyplot as plt
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer

# Disabilita la verifica SSL (utile per download di dataset in ambienti protetti)
ssl._create_default_https_context = ssl._create_unverified_context

# Imposta il seed per la riproducibilità
seed = 17
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Carica i dataframe di train e test
train_df = pd.read_csv('../Dataset/train_balanced_title.csv')
test_df = pd.read_csv('../Dataset/test_clean_title.csv')

# Se i file preprocessati non esistono, esegui il preprocessing e salvali
if not (os.path.exists('X_train.npz') and os.path.exists('X_test.npz')):
    # Preprocessing del train set
    x_train, y_train, scaler = libProject.doPreprocessing(train_df, scaler=None, fit_scaler=True)
    # Preprocessing del test set (usa lo stesso scaler)
    x_test, y_test, _ = libProject.doPreprocessing(test_df, scaler=scaler, fit_scaler=False)
    # Salva i dati preprocessati
    np.savez_compressed('X_train.npz', x=x_train, y=y_train)
    np.savez_compressed('X_test.npz', x=x_test, y=y_test)
else:
    # Carica i dati preprocessati
    train_data = np.load('X_train.npz')
    x_train, y_train = train_data['x'], train_data['y']
    test_data = np.load('X_test.npz')
    x_test, y_test = test_data['x'], test_data['y']

# Preprocessing della feature testuale
tfidf = TfidfVectorizer(max_features=50)  # Puoi cambiare max_features
X_train_title = tfidf.fit_transform(train_df['title_text'].fillna('')).toarray()
X_test_title = tfidf.transform(test_df['title_text'].fillna('')).toarray()

# Concatena le feature audio e testuali
x_train = np.concatenate([x_train, X_train_title], axis=1)
x_test = np.concatenate([x_test, X_test_title], axis=1)

# Crea i dataset PyTorch personalizzati
train_dataset = libProject.AudioFeaturesDataset(x_train, y_train)
test_dataset   = libProject.AudioFeaturesDataset(x_test, y_test)

# Parametri fissi per l'addestramento
epochs = 1000
batch_size = 256
lr = 0.0005
momentum = 0.95
weight_decay = 0.01
dropout = 0.5
hidden_units = 1024

# Crea i DataLoader per train e test
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Input size:", x_train.shape[1]) # Numero di feature in input
print("Output size:", len(np.unique(y_train))) # Numero di classi

# Nome del modello (include timestamp per unicità)
name = f"MLPClassifier_{int(time.time())}"
# Definisci la funzione di loss
loss = nn.CrossEntropyLoss()

# Istanzia il modello
model = libProject.MLPClassifier(x_train.shape[1], hidden_units, len(np.unique(y_train)), dropout=dropout)
# Allena il modello (early stopping incluso)
model = libProject.train_classifier(
    model, train_loader, test_loader, loss, exp_name=name, lr=lr, epochs=epochs, 
    momentum=momentum, weight_decay=weight_decay, early_stopping_patience=10
)

# Ottieni predizioni su train e test
predictions_train, labels_train = libProject.test_classifier(model, train_loader)
predictions_test, labels_test = libProject.test_classifier(model, test_loader)

# Carica metriche precedenti se esistono, altrimenti crea una nuova lista
if os.path.exists('metrics2.json'):
    with open('metrics2.json', 'r') as f:
        metrics = json.load(f)
else:
    metrics = []

# Aggiungi le metriche correnti
metrics.append({
    'model': name,
    'parameters': {
        'learning_rate': lr,
        'epochs': epochs,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'dropout': dropout,
        'hidden_units': hidden_units
    },
    'train_classification_report': classification_report(labels_train, predictions_train, output_dict=True),
    'test_classification_report': classification_report(labels_test, predictions_test, output_dict=True),
})

# Salva le metriche aggiornate su file
with open('metrics2.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Stampa le metriche dal file JSON
libProject.print_metrics_from_json('metrics2.json')