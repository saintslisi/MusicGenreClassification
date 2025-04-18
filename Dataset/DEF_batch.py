import pandas as pd
import os
import csv
import numpy as np
from MusicDownloader import download_youtube_audio
import ast

def ExtractFeatures(Tracks_csv_input = 'Tracks1.csv',features_csv_output = "TrackFeatures4.csv",Audio_path = "Tracks/"):
    """Versione 2.0 con matrici ridotte e rappresentazioni compatte della feature beats"""
    # File e cartelle
    print("🎧 Inizio elaborazione delle canzoni")
    print(Tracks_csv_input, features_csv_output, Audio_path)
    # Carica le canzoni da elaborare
    tracks_df = pd.read_csv(Tracks_csv_input, encoding="utf-16")
    print("OK")

    # Carica le feature già salvate, se esiste
    if os.path.exists(features_csv_output):
        features_df = pd.read_csv(features_csv_output)
        done_songs = set(features_df["id_track"].tolist())
    else:
        # Se il file non esiste, lo creiamo con l'header alla prima scrittura
        features_df = pd.DataFrame()
        done_songs = set()

    # Filtra le canzoni non ancora elaborate
    pending_tracks = tracks_df[~tracks_df["id_track"].isin(done_songs)]

    print(f"🎧 Canzoni totali: {len(tracks_df)}")
    print(f"✅ Già elaborate: {len(done_songs)}")
    print(f"🕐 Da elaborare: {len(pending_tracks)}")

    # Scrive nel file CSV
    for _, row in pending_tracks.iterrows():
        song_name = row["title"]
        genre = row.get("genre", "unknown")  # nel caso ci sia una colonna 'genre'

        print(f"\n🎵 Elaborazione: {song_name} ({genre})")

        try:
            # Scarica e ottieni le feature
            audio_features = download_youtube_audio(song_name, Audio_path,extract_sec=90)

            # Converte gli array numpy in liste per la scrittura nel CSV
            for key in audio_features:
                if hasattr(audio_features[key], "tolist"):
                    audio_features[key] = audio_features[key].tolist()

            # Crea un dizionario completo con ID e genere all'inizio
            features = {
                "id_track": row["id_track"],
                "genre": genre,
                **audio_features  # unisce le feature scaricate
            }

            # Se è la prima riga, scrivi header
            write_header = not os.path.exists(features_csv_output)

            with open(features_csv_output, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=features.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(features)

        except Exception as e:
            print(f"❌ Errore con '{song_name}': {e}")

# pathIN = "TestTracks.csv"
# pathOUT = "TestTracksFeatures.csv"
ExtractFeatures()