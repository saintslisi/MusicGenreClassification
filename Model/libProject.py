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

def string_to_dict(fd, key):
    """Converte una stringa in un dizionario Python"""
    print(f"\nConverto le stringhe di '{key}' contenenti mean e std in dizionari Python")
    parsed = fd[key].apply(ast.literal_eval)
    mean = parsed.apply(lambda x: x['mean'])
    std = parsed.apply(lambda x: x['std'])
    return mean, std

# Appiattisci una matrice 2D in un vettore 1D
def flatten_feature_matrix(matrix):
    return np.array(matrix).flatten()

i=0

# Funzione per il preprocessing di una singola riga
def preprocess_row(row):
    global i
    print(f"\nPreprocessing riga {i}...")

    i+=1
    try:
        mfccs = ast.literal_eval(row['mfccs'])
        chroma = ast.literal_eval(row['chroma'])
        spec_contrast = ast.literal_eval(row['spec_contrast'])
        zcr = ast.literal_eval(row['zcr'])
        beats = ast.literal_eval(row['beats'])
        tempo_feature = float(row['tempo'].replace("[", "").replace("]", ""))

        mfccs_vec = np.concatenate([
            flatten_feature_matrix(mfccs['mean']),
            flatten_feature_matrix(mfccs['std'])
        ])

        chroma_vec = np.concatenate([
            flatten_feature_matrix(chroma['mean']),
            flatten_feature_matrix(chroma['std'])
        ])

        spec_vec = np.concatenate([
            flatten_feature_matrix(spec_contrast['mean']),
            flatten_feature_matrix(spec_contrast['std'])
        ])

        zcr_vec = np.concatenate([
            flatten_feature_matrix(zcr['mean']),
            flatten_feature_matrix(zcr['std'])
        ])

        beats_vec = np.array([beats['count'], beats['interval_mean'], beats['interval_std']])

        # Unisci tutte le feature in un vettore
        return np.concatenate([mfccs_vec, chroma_vec, spec_vec, zcr_vec, [tempo_feature], beats_vec])
    except Exception as e:
        print(row)
        print(f"Errore nel preprocessing della riga {i}: {e}")
        return np.zeros(5944)  # Restituisci un vettore di zeri della lunghezza corretta in caso di errore

class AudioFeaturesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def doPreprocessing(fd):
    print("Preprocessing in corso...")
    feature_matrix = fd.apply(preprocess_row, axis=1)
    X = np.stack(feature_matrix.to_numpy())  # Convertiamo in matrice numpy

    # Label (target)
    y = fd[fd.columns[-1]]  # Etichetta dei generi musicali (ultima colonna)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler_sicuro.pkl") # Salva lo scaler

    # Se y è già 0...N-1, non serve LabelEncoder
    y_encoded = y.to_numpy()  # oppure np.array(y)
    print("Preprocessing completato.")
    return X_scaled, y_encoded

class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def add(self, value, num):
        self.sum += value*num
        self.count += num

    def value(self):
        try:
            return self.sum / self.count
        except:
            return None
        
def test_classifier(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions,labels = [], []
    with torch.no_grad():
        model.eval()
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            preds = output.to('cpu').max(1)[1].numpy()
            labs = y.to('cpu').numpy()
            predictions.extend(list(preds))
            labels.extend(list(labs))
    return np.array(predictions), np.array(labels)

def perc_error(gt, pred):
    return (1-accuracy_score(gt, pred))*100

def train_classifier(model, train_loader, test_loader, criterionL, exp_name="experiment", lr=0.001, epochs=10, momentum=0.99, weight_decay=0.001, logdir="logs"):

    if criterionL is None:
        criterionL = nn.CrossEntropyLoss()
    criterion = criterionL
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    #writer
    writer = SummaryWriter(join(logdir, exp_name))
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #definiamo un dizionario contenente i loader di training e testing
    loader = {
        "train": train_loader,
        "test": test_loader
    }
    #inizializziamo il global step
    global_step = 0

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        #iteriamo sui loader
        for mode in ["train", "test"]:
            #resettiamo i meter
            loss_meter.reset()
            acc_meter.reset()
            #settiamo il modello in training o evaluation
            if mode == "train":
                model.train()
            else:
                model.eval()
            #abilitiamo i gradienti solo in training
            with torch.set_grad_enabled(mode == "train"):
                for i, batch in enumerate(loader[mode]):
                    #prendiamo i dati e le etichette
                    x, y = batch
                    #spostiamo i dati sul device
                    x = x.to(device)
                    y = y.to(device)
                    #calcoliamo l'output del modello
                    output = model(x)
                    #aggiorniamo il global step
                    #conterrà il numero di campioni visti durante il training
                    n = x.shape[0]
                    global_step += n
                    #calcoliamo la loss
                    loss = criterion(output, y.long())
                    #se siamo in training facciamo il backward e l'ottimizzazione
                    if mode == "train":
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    #aggiorniamo il meter della loss
                    acc = accuracy_score(y.cpu(), output.max(1)[1].cpu())
                    loss_meter.add(loss.item(), n)
                    acc_meter.add(acc, n)
                    #loggiamo i risultati iterazione per iterazione solo in training
                    if mode == "train":
                        writer.add_scalar("loss/train", loss_meter.value(), global_step=global_step)
                        writer.add_scalar("accuracy/train", acc_meter.value(), global_step=global_step)
            #una volta finita l'epoca (sia nel caso di training che di testing) logghiamo i risultati
            writer.add_scalar("loss/"+mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar("accuracy/"+mode, acc_meter.value(), global_step=global_step)   
        #salviamo il modello
    torch.save(model.state_dict(), "models_weights/%s.pth" % (exp_name))   
    return model           

def print_metrics_from_json(json_path="metrics.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    for model_info in data:
        print(f"\nModel: {model_info.get('model', 'N/A')}")
        params = model_info.get("parameters", {})
        print(f"  learning_rate: {params.get('learning_rate', 'N/A')}")
        print(f"  epochs: {params.get('epochs', 'N/A')}")
        print(f"  momentum: {params.get('momentum', 'N/A')}")

        for split in ["train_classification_report", "test_classification_report"]:
            print(f"\n  {split}:")
            metrics = model_info.get(split, {})
            accuracy = metrics.get("accuracy", "N/A")
            print(f"    accuracy: {accuracy}")

            macro_avg = metrics.get("macro avg", {})
            weighted_avg = metrics.get("weighted avg", {})

            print("    macro avg:")
            for k, v in macro_avg.items():
                print(f"      {k}: {v}")

            print("    weighted avg:")
            for k, v in weighted_avg.items():
                print(f"      {k}: {v}")

# MODELLI

class LogisticRegressor(nn.Module):
    def __init__(self, in_size, out_size):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        return self.linear(x)

class SoftMaxRegressor(nn.Module):
    def __init__(self, in_size, out_size):
        super(SoftMaxRegressor, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        return self.linear(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, in_features, hidden_units, out_classes):
        super(MLPClassifier, self).__init__()
        self.hidden_layer = nn.Linear(in_features, hidden_units)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_units, out_classes)
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        return self.output_layer(x)
    
class DeepMLPClassifier(nn.Module):
    def __init__(self, in_features, hidden_units, out_classes, dropout = 0.5):
        super(DeepMLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_units, out_classes)
        )
    def forward(self, x):
        return self.model(x)