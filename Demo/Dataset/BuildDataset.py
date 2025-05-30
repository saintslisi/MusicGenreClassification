
"""
Script per l'estrazione automatica di metadati musicali da Spotify e la generazione
di un file CSV strutturato.

Funzionalità:
- Autenticazione tramite Spotify API utilizzando le credenziali definite in un file .env.
- Estrazione di brani da playlist associate a una lista predefinita di generi musicali.
- Raccolta delle informazioni: titolo del brano, artista, genere e indice locale.
- Costruzione di un dizionario strutturato con i metadati dei brani.
- Salvataggio dei dati in un file CSV (UTF-16) con intestazione.

Output:
- File CSV contenente i brani suddivisi per genere con le seguenti colonne:
  id_track, title, artist, genre, id_track_genre

Uso:
- Utile per la creazione di dataset musicali per analisi, visualizzazione o machine learning.

Nota:
- Limita il numero di brani per genere e playlist per evitare estrazioni eccessive.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Recupera le credenziali dall'ambiente
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("API_KEY")

# Configura l'autenticazione con Spotify
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Lista di generi musicali da esplorare
genres = [
    "Pop", "Rock", "Hip-Hop", "Jazz", "Blues", "Classical", "Electronic", "Reggae", "Funk", "Soul",
    "R&B", "Metal", "Punk", "Folk", "Country", "Disco", "Techno", "House", "Trance", "Drum and Bass",
    "Dubstep", "Ambient", "Indie", "Alternative", "Latin", "K-Pop", "Afrobeat", "Gospel", "Ska", "Grunge"
]

brani_generi = []
titles = []

countBrani = 0
TotBrani = 4
TotTrackForGen = (len(genres) / TotBrani) + 1
end = False

# Ciclo principale sui generi
for _, gen in enumerate(genres):
    print(f"{_ + 1}) Genere: {gen}")
    count = 0
    countTrackGen = 0

    if end:
        break

    # Cerco playlist su Spotify per ogni genere
    for __, res in enumerate(sp.search(q=f"{gen}", type='playlist')['playlists']['items']):
        if res is not None:
            if count == 4:
                break

            totCanz = res['tracks']['total']
            if totCanz > 100:
                totCanz = 99

            count += 1
            playlistTracks = sp.playlist_tracks(res['id'], limit=totCanz)

            for ___, track in enumerate(playlistTracks['items']):
                if countTrackGen == TotTrackForGen:
                    end = True
                    break

                if track and track['track']:
                    track_name = track['track']['name']
                    artist_name = track['track']['artists'][0]['name']

                    if track_name not in titles:
                        if artist_name:
                            # Formatta il nome del brano evitando simboli problematici
                            track_clean = track_name.replace(":", "").replace("_", "").replace(",", "")
                            artist_clean = artist_name.replace(":", "").replace("_", "").replace(",", "")
                            brani_generi.append(f"{gen}:{track_clean}_{artist_clean}_{countTrackGen}")

                            countBrani += 1
                            countTrackGen += 1
                            titles.append(track_name)

# Stampa riepilogativa dei brani raccolti
print("\n\n\n\nBrani Generi:")
for _, brano in enumerate(brani_generi):
    print(f"{_ + 1}) {brano}")
print("Totale:", len(brani_generi))

# Costruzione dizionario con metadati delle tracce
featuresTracks = {}
for _ in range(len(brani_generi)):
    parts = brani_generi[_].split(":")[1].split("_")
    genre = brani_generi[_].split(":")[0]
    title = parts[0]
    artist = parts[1]
    id_track_genre = parts[2]

    featuresTracks[f'{_}'] = {
        "id_track": _,
        "title": f"{title} {artist}",
        "artist": artist,
        "genre": genre,
        "id_track_genre": id_track_genre,
    }

# Salvataggio su file CSV
csv = open("Dataset/Tracks.csv", "w", encoding="utf-16")
csv.write("id_track,title,artist,genre,id_track_genre\n")

for _, track in enumerate(featuresTracks):
    try:
        csv.write(f"{featuresTracks[track]['id_track']},{featuresTracks[track]['title']},"
                  f"{featuresTracks[track]['artist']},{featuresTracks[track]['genre']},"
                  f"{featuresTracks[track]['id_track_genre']}\n")
    except Exception as e:
        print(f"Errore con {featuresTracks[track]['title']}: {e}")
        continue

print("\n\nTotale:", len(brani_generi))
print(f"IN TITLES: {len(titles)}")
print("\nEND")
