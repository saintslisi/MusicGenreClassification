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
ssl._create_default_https_context = ssl._create_unverified_context

# fd = pd.read_csv('../Dataset/dataset_macro_generi.csv')

# X_scaled, y_encoded = libProject.doPreprocessing(fd)

# # Salva X_scaled e y_encoded in un file JSON
# with open('preprocessed_data_macro.json', 'w') as f_json:
#     json.dump({
#         'X_scaled': X_scaled.tolist(),
#         'y_encoded': y_encoded.tolist()
#     }, f_json)

# Carica X_scaled e y_encoded dal file JSON se non presente il file crearlo con il codice commentato sopra
with open('preprocessed_data_macro_balanced.json', 'r') as f_json:
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
name = f"MLPClassifierHU1024_{int(time.time())}"
loss = nn.CrossEntropyLoss()
lr = 0.01
epochs = 100
momentum = 0.9
wight_decay = 0.0001
dropout = 0.5
#e cambialo anche qui
model = libProject.MLPClassifier(x_train.shape[1], 1024, len(np.unique(y_train)))
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
        'batch_size': 1024,
        #'dropout': dropout
    },
    'train_classification_report': classification_report(labels_train, predictions_train, output_dict=True),
    'test_classification_report': classification_report(labels_test, predictions_test, output_dict=True),
})

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

libProject.print_metrics_from_json('metrics.json')


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
