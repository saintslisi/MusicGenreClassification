import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from count import count_songs_by_genre

class MusicGenreClassifier(nn.Module):
    def __init__(self):
        super(MusicGenreClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5944, 1000),
            nn.ReLU(),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            #nn.ReLU(),
            #nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 21)
        )

    def forward(self, x):
        return self.net(x)
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


fd = pd.read_csv('TestTracks.csv', encoding="utf-16")
dim = fd.shape[0]

scaler = joblib.load("scaler_sicuro.pkl")
pathOUT = "TestTracksFeatures.csv"
y = ['Reggae', 'Jazz', 'Funk', 'Soul', 'House', 'Dubstep', 'Electronic', 'Blues',
        'Disco', 'Pop', 'Classical', 'Hip-Hop', 'Drum and Bass', 'R&B', 'Techno',
        'Country', 'Folk', 'Punk', 'Metal', 'Trance', 'Rock']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # y → array di interi
joblib.dump(scaler, "labelEncoder_sicuro.pkl")

x = 0
st.write("Scegli un elemento dallo slider")
x = st.slider('',0,dim)  # 👈 this is a widget   
f = f"'{fd.iloc[x]['title']}' di {fd.iloc[x]['artist']}"
st.write(f)

import torch.nn.functional as F

if st.button("Predici il genere musicale"):
    model = MusicGenreClassifier()
    model.load_state_dict(torch.load("modello_sicuro.pth"))
    test_df = pd.read_csv(pathOUT)
    new_feature_vector = preprocess_row(test_df.iloc[x])  # stessa funzione usata prima
    new_feature_vector = scaler.transform([new_feature_vector])  # normalizzazione  
    new_tensor = torch.tensor(new_feature_vector, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        output = model(new_tensor)
        probabilities = F.softmax(output, dim=1).numpy()[0]  # Applica softmax e ottieni l'array 1D

    #print(f"Probabilità: {probabilities}")
    top3_indices = np.argsort(probabilities)[-3:][::-1]  # Ordina e prendi i primi 3 (in ordine decrescente)

    # Mappa gli indici ai nomi dei generi
    predicted_genres = label_encoder.inverse_transform(top3_indices)

    # Mostra i risultati con le probabilità
    #print(f"🎧 Genere musicale predetto di: '{fd.iloc[x]['title']}': {predicted_genres}")    
    for genre, prob in zip(predicted_genres, probabilities[top3_indices]):
        st.write(f"\t{genre}: {prob*100:.2f}%")

st.write("Conteggio generi")
if st.button("Vedi quanti generi sono presenti nel dataset"):
    g,t = count_songs_by_genre('Tracks1.csv')
    st.write(f"Totale canzoni: {t}")
    st.write("Numero di canzoni per genere:")
    st.write(g)
    #st.write("Controlla la console per i risultati")