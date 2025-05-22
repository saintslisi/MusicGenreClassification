import libProject # libreria del progetto
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report
from os.path import join
import time
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import ssl
import json
ssl._create_default_https_context = ssl._create_unverified_context

# fd = pd.read_csv('../Dataset/TrackFeatures4.csv')

# X_scaled, y_encoded = libProject.doPreprocessing(fd)

# # Salva X_scaled e y_encoded in un file JSON
# with open('preprocessed_data.json', 'w') as f_json:
#     json.dump({
#         'X_scaled': X_scaled.tolist(),
#         'y_encoded': y_encoded.tolist()
#     }, f_json)

# Carica X_scaled e y_encoded dal file JSON se non presente il file crearlo con il codice commentato sopra
with open('preprocessed_data.json', 'r') as f_json:
    data = json.load(f_json)
    X_scaled = np.array(data['X_scaled'])
    y_encoded = np.array(data['y_encoded'])

#distribuzione in train e test
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=123, stratify=y_encoded
)

# Crea i dataset
train_dataset = libProject.AudioFeaturesDataset(x_train, y_train)
test_dataset   = libProject.AudioFeaturesDataset(x_test, y_test)

# Crea i DataLoader
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

print("Input size:", x_train.shape[1])
print("Output size:", len(np.unique(y_train)))

#cambia modello qui
name = f"DeepMLPClassifierHU64_{int(time.time())}"
loss = nn.CrossEntropyLoss()
lr = 0.01
epochs = 1000
momentum = 0.99
wight_decay = 0.001
#e cambialo anche qui
model = libProject.DeepMLPClassifier(x_train.shape[1], 64, len(np.unique(y_train)))
model = libProject.train_classifier(model, train_loader, test_loader, loss, exp_name=name, lr=lr, epochs=epochs, momentum=momentum, weight_decay=wight_decay)

predictions_train, labels_train = libProject.test_classifier(model, train_loader)
predictions_test, labels_test = libProject.test_classifier(model, test_loader)

with open('metrics.json', 'r') as f:
    metrics = json.load(f)

metrics.append({
    'model': name,
    'parameters': {
        'learning_rate': lr,
        'epochs': epochs,
        'momentum': momentum,
        'weight_decay': wight_decay,
        'loss' : str(loss)
    },
    'train_classification_report': classification_report(labels_train, predictions_train, output_dict=True),
    'test_classification_report': classification_report(labels_test, predictions_test, output_dict=True),
})

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

libProject.print_metrics_from_json('metrics.json')