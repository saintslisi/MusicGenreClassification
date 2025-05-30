import pandas as pd
import os
import csv

# Lista dei 9 generi accettati
accepted_genres = [
    "Ambient/Other",
    "Classical",
    "Electronic/House",
    "Folk/Country",
    "Hip-Hop",
    "Jazz/Blues",
    "Pop",
    "Reggae/Afrobeat",
    "Rock"
]
def insertData(data, titoloArtista, artista, path="Demo/Modello/"):
    """
    Inserisce un nuovo brano nei CSV dei metadati e delle feature, 
    solo se non è già presente (match su titolo + artista).

    Se già presente, restituisce il suo ID esistente.
    Altrimenti, salva i dati e restituisce il nuovo ID assegnato.
    """
    for key in data:
        if isinstance(data[key], str):
            data[key] = data[key].replace("nan", "0")
        elif hasattr(data[key], "tolist"):
            data[key] = data[key].tolist()
        elif isinstance(data[key], float) and str(data[key]) == "nan":
            data[key] = 0.0
    # Carica dataset informazioni e feature
    info_path = f'{path}TestTracks.csv'
    feat_path = f'{path}TestTracksFeatures.csv'

    fd_Info = pd.read_csv(info_path, encoding="utf-16")
    fd_Feature = pd.read_csv(feat_path, encoding="utf-8")

    # ✅ Verifica se già presente (titolo + artista)
    match = fd_Info[(fd_Info["title"] == titoloArtista) & (fd_Info["artist"] == artista)]

    if not match.empty:
        existing_id = int(match.iloc[0]["id_track"])
        print(f"✅ Brano già presente con ID: {existing_id}")
        return existing_id-1

    # 🔧 Se non presente, prepara i dati da inserire
    for key in data:
        if hasattr(data[key], "tolist"):
            data[key] = data[key].tolist()

    new_id = len(fd_Feature) + 1


    genres = get_genres_by_artist_and_title(artista,titoloArtista)
    info = {
        "id_track": new_id,
        "title": titoloArtista,
        "artist": artista,
        "genre": genres,
        "id_track_genre": new_id
    }

    # 📥 Aggiungi nuove righe ai DataFrame
    id_to_label = {
        0: "Ambient/Other",
        1: "Classical",
        2: "Electronic",
        3: "Folk/Country",
        4: "Hip-Hop",
        5: "Jazz/Blues",
        6: "Pop",
        7: "Reggae/Afrobeat",
        8: "Rock"
    }
    features = {
        "id_track": new_id,
        "genre": genres[0],
        **data,
        "macro_genre": id_to_label.get(data.get("macro_genre_id", 8), "Rock"),  # Default to Rock if not found
        "macro_genre_id": data.get("macro_genre_id", 8)  # Default to 8 if not found
    }    

    # Scrivi la riga nel file (aggiunge header se il file è nuovo)
    write_header = not os.path.exists(feat_path)
    with open(feat_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=features.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(features)    
    
    df_info = pd.concat([fd_Info, pd.DataFrame([info])], ignore_index=True)

    # 💾 Salva i file aggiornati
    #df_feat.to_csv(feat_path, index=False, encoding="utf-8")
    df_info.to_csv(info_path, index=False, encoding="utf-16")

    print(f"✅ Nuovo brano inserito con ID: {new_id}")
    return new_id-1

import musicbrainzngs

musicbrainzngs.set_useragent("MyMusicApp", "1.0", "youremail@example.com")

def get_genres_by_artist_and_title(artist_name, track_title):
    try:
        # Cerca registrazioni che corrispondono a artista + titolo
        result = musicbrainzngs.search_recordings(artist=artist_name, recording=track_title, limit=1)
        
        if not result['recording-list']:
            print(f"Nessuna registrazione trovata per '{track_title}' di '{artist_name}'")
            return ["Sconosciuto"]
        
        recording = result['recording-list'][0]
        
        # Ottieni artista principale (primo artist-credit)
        artist_credit = recording.get('artist-credit', [])
        if not artist_credit:
            print("Nessun artista trovato nella registrazione")
            return ["Sconosciuto"]
        
        main_artist = artist_credit[0].get('artist', {})
        artist_id = main_artist.get('id')
        if not artist_id:
            print("ID artista non trovato")
            return ["Sconosciuto"]
        
        # Ora recupera i dettagli dell'artista per i tag (generi)
        artist_info = musicbrainzngs.get_artist_by_id(artist_id, includes=["tags"])
        artist_data = artist_info.get('artist', {})
        
        tags = artist_data.get('tag-list', [])
        if not tags:
            print(f"Nessun genere trovato per l'artista '{main_artist.get('name', '')}'")
            return ["Sconosciuto"]
        
        genres = [tag['name'] for tag in tags]
        lista_minuscole = [s.lower() for s in accepted_genres]
        print(genres) 
        resGen = [] 
        for gen in genres:
            for g in lista_minuscole:
                if gen in g:
                    print(f"{gen} è in lista")
                    resGen.append(g)
        lista_formattata = []
        if len(resGen) == 0:
            lista_formattata.append("Sconosciuto")
        else:
            lista_formattata = [s.title() for s in set(resGen)]
        print(lista_formattata)
        return lista_formattata

    except Exception as e:
        return f"Errore: {e}"

# Esempio d'uso
titolo = "Livin'la Vida Loca"
artista = "Ricky Martin"
generi = get_genres_by_artist_and_title(artista, titolo)
print(f"Generi per '{titolo}' di '{artista}': {generi}")


