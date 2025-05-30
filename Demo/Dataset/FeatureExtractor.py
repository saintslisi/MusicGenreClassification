"""
Script per l'estrazione e la compattazione di feature audio da file MP3.

Funzionalità principali:
- Carica un file audio e ne estrae le principali feature temporali e spettrali.
- Calcola le medie e deviazioni standard per ciascuna feature su finestre temporali di durata specificata.
- Restituisce le seguenti feature in formato compatto:
  - MFCC (Mel Frequency Cepstral Coefficients)
  - Chroma STFT
  - Spectral Contrast
  - Zero-Crossing Rate
  - Tempo
  - Statistiche sui battiti (numero, intervallo medio, deviazione standard)

Output:
- Un dizionario Python contenente tutte le feature, già pronte per essere salvate o usate in modelli ML.

Utilizzo:
- Utile per la preparazione di dataset audio in progetti di classificazione musicale, riconoscimento del genere o analisi del contenuto audio.

Nota:
- È possibile regolare la durata dell’analisi (parametro `sec`) per controllare la granularità temporale del riassunto.
"""


import librosa
import numpy as np

"""
Versione 3.0: Estrazione feature audio da file MP3.
- Le feature MFCC, Chroma, Spectral Contrast e ZCR vengono compattate in matrici (mean + std).
- Tempo e Beats vengono riassunti ma non compressi in matrici.
"""

def summarize_matrix(matrix, sec=30):
    """
    Restituisce due matrici (mean e std) di dimensione (n_features, sec).
    Ogni colonna rappresenta media/deviazione standard delle feature in ciascun secondo.
    Sostituisce i NaN con 0.
    """
    print(f"\n🎧 Compattamento matrice {matrix.shape} in {sec} secondi")

    means = []
    stds = []

    for i in range(sec):
        start = i * sec
        end = start + sec
        chunk = matrix[:, start:end]  # Seleziona la finestra temporale
        means.append(np.mean(chunk, axis=1))
        stds.append(np.std(chunk, axis=1))

    # Stack per ottenere matrici finali (n_feature x sec)
    mean_matrix = np.stack(means, axis=1)
    std_matrix = np.stack(stds, axis=1)

    # Sostituisci i NaN con 0
    mean_matrix = np.nan_to_num(mean_matrix, nan=0.0)
    std_matrix = np.nan_to_num(std_matrix, nan=0.0)

    print(f"✔️ Shape output: mean {mean_matrix.shape}, std {std_matrix.shape}")

    return {
        "mean": mean_matrix.tolist(),
        "std": std_matrix.tolist()
    }
def extract_features(path, sec=30) -> dict:
    """
    Estrae e compattata le seguenti feature da un file audio:
    - MFCC
    - Chroma
    - Spectral Contrast
    - Zero-Crossing Rate
    - Tempo
    - Beat statistics (count, mean interval, std interval)
    """
    print(f"\n\n🎧 Caricamento audio da: {path} per {sec} secondi")
    y, sr = librosa.load(path, sr=None)
    print("✔️ Audio caricato")

    # 1. MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_summary = summarize_matrix(mfccs, sec)

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_summary = summarize_matrix(chroma, sec)

    # 3. Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_summary = summarize_matrix(spec_contrast, sec)

    # 4. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_summary = summarize_matrix(zcr, sec)

    # 5. Tempo e Beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_intervals = np.diff(beats)
    mean_interval = np.mean(beat_intervals)
    std_interval = np.std(beat_intervals)



    return {
        "mfccs": mfccs_summary,
        "chroma": chroma_summary,
        "spec_contrast": spec_contrast_summary,
        "zcr": zcr_summary,
        "tempo": tempo,
        "beats": {
            "count": len(beats),
            "interval_mean": mean_interval,
            "interval_std": std_interval
        }
    }


#extract_features(f"Demo/Track/tmp90.mp3", sec=90)