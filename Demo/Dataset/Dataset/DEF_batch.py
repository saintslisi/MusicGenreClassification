"""
Script per l'estrazione batch delle feature audio da una lista di brani (dataset CSV).

Funzionalità:
- Carica un file CSV con una lista di brani (titolo, id_track, genere).
- Per ogni brano non ancora elaborato:
    - Scarica l’audio da YouTube.
    - Estrae le feature centrali (MFCC, Chroma, Spectral Contrast, ZCR, Tempo, Beats).
- Le feature vengono convertite e salvate in un nuovo file CSV compatto.

Input:
- `Tracks_csv_input`: file con i metadati dei brani.
- `features_csv_output`: file CSV di output con le feature.
- `Audio_path`: cartella dove scaricare e salvare gli MP3 temporanei.

Dipendenze:
- pandas, numpy, os, csv
- Modulo personalizzato `MusicDownloader` per scaricare e processare l'audio.

Nota:
- Riconosce automaticamente i brani già processati per evitare duplicati.
- Le feature sono rappresentate in forma compatta (es. mean/std).
"""

import pandas as pd
import os
import csv
from MusicDownloader import download_youtube_audio


def ExtractFeatures(
    Tracks_csv_input='Dataset/Tracks1.csv',
    features_csv_output='Dataset/TrackFeatures.csv',
    Audio_path='Dataset/Tracks/'
):
    print("🎧 Inizio elaborazione delle canzoni")
    print(f"📂 Input: {Tracks_csv_input}, Output: {features_csv_output}, Path audio: {Audio_path}")

    # Carica i metadati dei brani
    tracks_df = pd.read_csv(Tracks_csv_input, encoding="utf-16")
    print("✔️ File dei brani caricato.")

    # Carica le feature già elaborate se esistono
    if os.path.exists(features_csv_output):
        features_df = pd.read_csv(features_csv_output)
        done_songs = set(features_df["id_track"].tolist())
    else:
        features_df = pd.DataFrame()
        done_songs = set()

    # Seleziona i brani ancora da elaborare
    pending_tracks = tracks_df[~tracks_df["id_track"].isin(done_songs)]

    print(f"🎧 Canzoni totali: {len(tracks_df)}")
    print(f"✅ Già elaborate: {len(done_songs)}")
    print(f"🕐 Da elaborare: {len(pending_tracks)}")

    # Estrazione e salvataggio delle feature
    for _, row in pending_tracks.iterrows():
        song_name = row["title"]
        genre = row.get("genre", "unknown")

        print(f"\n🎵 Elaborazione: {song_name} ({genre})")

        try:
            # Scarica l’audio ed estrae le feature
            audio_features = download_youtube_audio(song_name, Audio_path, extract_sec=90)

            # Converti array numpy in liste per serializzazione
            for key in audio_features:
                if hasattr(audio_features[key], "tolist"):
                    audio_features[key] = audio_features[key].tolist()

            # Prepara dizionario da scrivere nel CSV
            features = {
                "id_track": row["id_track"],
                "genre": genre,
                **audio_features
            }

            # Scrivi la riga nel file (aggiunge header se il file è nuovo)
            write_header = not os.path.exists(features_csv_output)
            with open(features_csv_output, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=features.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(features)

        except Exception as e:
            print(f"❌ Errore con '{song_name}': {e}")


# Esecuzione diretta
#ExtractFeatures()
