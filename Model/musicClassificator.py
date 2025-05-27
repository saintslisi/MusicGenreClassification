import libProject # libreria del progetto
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from os.path import join
import time
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import ssl
import json
import matplotlib.pyplot as plt
import os
import random
ssl._create_default_https_context = ssl._create_unverified_context

seed = 17
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_df = pd.read_csv('../Dataset/train_balanced.csv')
test_df = pd.read_csv('../Dataset/test_clean.csv')

if not (os.path.exists('X_train.npz') and os.path.exists('X_test.npz')):
    # Preprocessing train
    x_train, y_train, scaler = libProject.doPreprocessing(train_df, scaler=None, fit_scaler=True)
    # Preprocessing test (usa lo stesso scaler)
    x_test, y_test, _ = libProject.doPreprocessing(test_df, scaler=scaler, fit_scaler=False)
    np.savez_compressed('X_train.npz', x=x_train, y=y_train)
    np.savez_compressed('X_test.npz', x=x_test, y=y_test)
else:
    train_data = np.load('X_train.npz')
    x_train, y_train = train_data['x'], train_data['y']
    test_data = np.load('X_test.npz')
    x_test, y_test = test_data['x'], test_data['y']

# #distribuzione in train e test
# x_train, x_test, y_train, y_test = train_test_split(
#     X_scaled, y_encoded, test_size=0.2, random_state=123, stratify=y_encoded
# )

# Crea i dataset
train_dataset = libProject.AudioFeaturesDataset(x_train, y_train)
test_dataset   = libProject.AudioFeaturesDataset(x_test, y_test)

with open('par.json', 'r') as f:
    param_grid = json.load(f)

for i, values in enumerate(param_grid):
    print(f"Training model with parameters set {i+1}")
    epochs = 1000
    batch_size = values.get('batch_size', 256)
    lr = values.get('learning_rate', 0.001)
    momentum = values.get('momentum', 0.95)
    wight_decay = values.get('weight_decay', 0.0005)
    dropout = values.get('dropout', 0.5)
    hidden_units = values.get('hidden_units', 1024)

    # Crea i DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Input size:", x_train.shape[1]) #5944
    print("Output size:", len(np.unique(y_train))) #9

    #cambia modello qui
    name = f"DeepMLPClassifier_{int(time.time())}"
    loss = nn.CrossEntropyLoss()

    #e cambialo anche qui
    model = libProject.DeepMLPClassifier(x_train.shape[1], hidden_units, len(np.unique(y_train)), dropout=dropout)
    model = libProject.train_classifier(model, train_loader, test_loader, loss, exp_name=name, lr=lr, epochs=epochs, momentum=momentum, weight_decay=wight_decay, early_stopping_patience=10)

    predictions_train, labels_train = libProject.test_classifier(model, train_loader)
    predictions_test, labels_test = libProject.test_classifier(model, test_loader)

    if os.path.exists('metrics2.json'):
        with open('metrics2.json', 'r') as f:
            metrics = json.load(f)
    else:
        metrics = []

    metrics.append({
        'model': name,
        'parameters': {
            'learning_rate': lr,
            'epochs': epochs,
            'momentum': momentum,
            'weight_decay': wight_decay,
            'batch_size': batch_size,
            'dropout': dropout,
            'hidden_units': hidden_units
        },
        'train_classification_report': classification_report(labels_train, predictions_train, output_dict=True),
        'test_classification_report': classification_report(labels_test, predictions_test, output_dict=True),
    })

    with open('metrics2.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    libProject.print_metrics_from_json('metrics2.json')


# # Mappa ID → nome del macro-genere (in ordine!)
# id_to_label = {
#     0: "Ambient/Other",
#     1: "Classical",
#     2: "Electronic",
#     3: "Folk/Country",
#     4: "Hip-Hop",
#     5: "Jazz/Blues",
#     6: "Pop",
#     7: "Reggae/Afrobeat",
#     8: "Rock"
# }

# # Ordine delle etichette (display_labels deve essere una lista ordinata per ID)
# labels = [id_to_label[i] for i in sorted(id_to_label)]

# # Crea matrice di confusione
# cm = confusion_matrix(labels_test, predictions_test)

# # Mostra la matrice
# fig, ax = plt.subplots(figsize=(12, 12))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
# plt.title("Matrice di Confusione")
# plt.show()
