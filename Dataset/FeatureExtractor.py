import librosa
import numpy as np
"""Versione 3.0 con matrici ridotte e rappresentazioni compatte della feature beats
matrici semi ridotte più o meno nXm (n: n_feature) (m: secondi di mean e std)"""
def summarize_matrix(matrix, sec=30):
    """
    Restituisce due matrici (mean, std) di dimensione (n_features, n_sec),
    dove n_sec = durata audio in secondi.
    Ogni colonna rappresenta la media o deviazione standard delle feature nel secondo corrispondente.
   """

    print(f"\n🎧 Compattamento matrice {matrix.shape} in {sec} secondi")
    means = []
    stds = []

    for i in range(sec):
        start = i * sec
        end = start + sec

        chunk = matrix[:, start:end]  # shape: (n_features, frames_in_secondo)
        means.append(np.mean(chunk, axis=1))
        stds.append(np.std(chunk, axis=1))

    mean_matrix = np.stack(means, axis=1)  # shape: (n_features, total_seconds)
    std_matrix = np.stack(stds, axis=1)
    #print(f"\n\n🎧 Matrici MFCC, Chroma, Spectral Contrast e ZCR compattate in {sec} secondi")
    print(mean_matrix.shape, std_matrix.shape)
    return {
        "mean": mean_matrix.tolist(),
        "std": std_matrix.tolist()
    }


def extract_features(path, sec=30) -> dict:
    """
    Estrae e compatta le feature audio da un file MP3:
    - MFCC, Chroma, Spectral contrast, ZCR -> compressi
    - Tempo & Beats -> invariati
    """
    print(f"\n\n\n🎧 Caricamento audio da: {path} di {sec} sec")
    y, sr = librosa.load(path,sr=None)
    print("🎧 Audio caricato")

    # 1. MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,)
    mfccs_summary = summarize_matrix(mfccs,sec)

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_summary = summarize_matrix(chroma,sec)

    # 3. Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_summary = summarize_matrix(spec_contrast,sec)

    # 4. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_summary = summarize_matrix(zcr,sec)

    # 5. Tempo & Beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats_count = len(beats)
    
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
            "count": beats_count,
            "interval_mean": mean_interval,
            "interval_std": std_interval
            }
    }

def extract_features2(path, sec=30) -> dict:
    """
    Estrae le feature audio da un file MP3 senza compattarle/summarizzarle:
    - MFCC, Chroma, Spectral contrast, ZCR -> matrici complete
    - Tempo & Beats -> invariati
    """
    print(f"\n\n\n🎧 Caricamento audio da: {path} di {sec} sec")
    y, sr = librosa.load(path, sr=None)
    print("🎧 Audio caricato")

    # 1. MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 3. Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # 4. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # 5. Tempo & Beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats_count = len(beats)
    beat_intervals = np.diff(beats)
    mean_interval = np.mean(beat_intervals) if beat_intervals.size > 0 else 0
    std_interval = np.std(beat_intervals) if beat_intervals.size > 0 else 0

    return {
        "mfccs": mfccs.tolist(),
        "chroma": chroma.tolist(),
        "spec_contrast": spec_contrast.tolist(),
        "zcr": zcr.tolist(),
        "tempo": tempo,
        "beats": {
            "count": beats_count,
            "interval_mean": mean_interval,
            "interval_std": std_interval
        }
    }
