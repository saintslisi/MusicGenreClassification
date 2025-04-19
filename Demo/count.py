import pandas as pd

def count_songs_by_genre(tracks_csv):
    # Carica il CSV
    tracks_df = pd.read_csv(tracks_csv, encoding="utf-16")
    
    # Calcola il totale per ogni genere
    genre_counts = tracks_df["genre"].value_counts()
    
    # Calcola il totale complessivo delle canzoni
    total_songs = len(tracks_df)
    
    # Stampa il conteggio per ogni genere
    print(f"🎵 Totale canzoni: {total_songs}")
    print("🎶 Numero di canzoni per genere:")
    genres = []
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")
        genres.append(genre)
    print(f"Totale generi: {len(genres)}\n{genres}")
    return genre_counts, total_songs


# Usa la funzione
#count_songs_by_genre('Tracks.csv')

# print("\n\n\n\n")
# df = pd.read_csv("TrackFeatures.csv")

# import ast
# import numpy as np

# mfcc = np.array(ast.literal_eval(df["mfccs"].iloc[0]))
# print(mfcc.shape)
