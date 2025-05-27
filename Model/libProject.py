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
    
def doPreprocessing(fd, scaler=None, fit_scaler=True):
    print("Preprocessing in corso...")
    feature_matrix = fd.apply(preprocess_row, axis=1)
    X = np.stack(feature_matrix.to_numpy())
    y = fd[fd.columns[-1]]

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler_sicuro.pkl")
    else:
        X_scaled = scaler.transform(X)

    y_encoded = y.to_numpy()
    print("Preprocessing completato.")
    return X_scaled, y_encoded, scaler

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
        
def top_k_accuracy(output, target, k=3):
    """Restituisce la top-k accuracy per un batch."""
    with torch.no_grad():
        # output: [batch_size, num_classes]
        # target: [batch_size]
        topk = output.topk(k, dim=1).indices  # [batch_size, k]
        # Confronta se il target è tra i top-k
        correct = topk.eq(target.unsqueeze(1)).sum().item()
        return correct / target.size(0)
        
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

def train_classifier(model, train_loader, test_loader, criterionL, exp_name="experiment", lr=0.001, epochs=10, momentum=0.99, weight_decay=0.001, logdir="logs2", early_stopping_patience=10):
    if criterionL is None:
        criterionL = nn.CrossEntropyLoss()
    criterion = criterionL
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    acc3_meter = AverageValueMeter()
    writer = SummaryWriter(join(logdir, exp_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = {"train": train_loader, "test": test_loader}
    global_step = 0

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        for mode in ["train", "test"]:
            loss_meter.reset()
            acc_meter.reset()
            acc3_meter.reset()
            if mode == "train":
                model.train()
            else:
                model.eval()
            with torch.set_grad_enabled(mode == "train"):
                for i, batch in enumerate(loader[mode]):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                    n = x.shape[0]
                    global_step += n
                    loss = criterion(output, y.long())
                    if mode == "train":
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    acc = accuracy_score(y.cpu(), output.max(1)[1].cpu())
                    top3_acc = top_k_accuracy(output, y, k=3)
                    loss_meter.add(loss.item(), n)
                    acc_meter.add(acc, n)
                    acc3_meter.add(top3_acc, n)
                    if mode == "train":
                        writer.add_scalar("loss/train", loss_meter.value(), global_step=global_step)
                        writer.add_scalar("accuracy/train", acc_meter.value(), global_step=global_step)
                        writer.add_scalar("top3_accuracy/train", acc3_meter.value(), global_step=global_step)
            writer.add_scalar("loss/"+mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar("accuracy/"+mode, acc_meter.value(), global_step=global_step)
            writer.add_scalar("top3_accuracy/"+mode, acc3_meter.value(), global_step=global_step)

        # Early stopping check (dopo ogni epoca)
        val_loss = loss_meter.value()  # loss_meter contiene la loss dell'ultimo mode (test)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Salva il modello migliore
            torch.save(model.state_dict(), "models_weights/%s_best.pth" % (exp_name))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {e+1}")
                # Carica il modello migliore prima di uscire
                model.load_state_dict(torch.load("models_weights/%s_best.pth" % (exp_name)))
                return model

    torch.save(model.state_dict(), "models_weights/%s.pth" % (exp_name))
    return model          

def print_metrics_from_json(json_path="metrics.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Per trovare i migliori modelli
    best_acc = -1
    best_acc_model = None
    best_f1_macro = -1
    best_f1_macro_model = None
    best_f1_weighted = -1
    best_f1_weighted_model = None

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

        # Trova i migliori modelli sulla base del test set
        test_metrics = model_info.get("test_classification_report", {})
        test_acc = test_metrics.get("accuracy", -1)
        test_f1_macro = test_metrics.get("macro avg", {}).get("f1-score", -1)
        test_f1_weighted = test_metrics.get("weighted avg", {}).get("f1-score", -1)

        if test_acc is not None and test_acc > best_acc:
            best_acc = test_acc
            best_acc_model = model_info.get('model', 'N/A')
        if test_f1_macro is not None and test_f1_macro > best_f1_macro:
            best_f1_macro = test_f1_macro
            best_f1_macro_model = model_info.get('model', 'N/A')
        if test_f1_weighted is not None and test_f1_weighted > best_f1_weighted:
            best_f1_weighted = test_f1_weighted
            best_f1_weighted_model = model_info.get('model', 'N/A')

    print("\n--- Migliori modelli sul test set ---")
    print(f"Test accuracy più alta: {best_acc_model} ({best_acc})")
    print(f"Test f1 macro più alta: {best_f1_macro_model} ({best_f1_macro})")
    print(f"Test f1 weighted più alta: {best_f1_weighted_model} ({best_f1_weighted})")

# MODELLI

class SoftMaxRegressor(nn.Module):
    def __init__(self, in_size, out_size):
        super(SoftMaxRegressor, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        return self.linear(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, in_features, hidden_units, out_classes, dropout=0.3):
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
    
class DeepMLPClassifier(nn.Module):
    def __init__(self, in_features, hidden_units, out_classes, dropout = 0.3):
        super(DeepMLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, int(hidden_units/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_units/2), out_classes)
        )
    def forward(self, x):
        return self.model(x)