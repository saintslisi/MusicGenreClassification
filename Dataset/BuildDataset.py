import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
from pprint import pprint

from MusicDownloader import download_youtube_audio
# Carica le variabili d'ambiente dal file .env (se lo usi)
load_dotenv()

# Recupera le credenziali dall'ambiente
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("API_KEY")

# Configura l'autenticazione
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

genres = [
    "Pop", "Rock", "Hip-Hop", "Jazz", "Blues", "Classical", "Electronic", "Reggae", "Funk", "Soul",
    "R&B", "Metal", "Punk", "Folk", "Country", "Disco", "Techno", "House", "Trance", "Drum and Bass",
    "Dubstep", "Ambient", "Indie", "Alternative", "Latin", "K-Pop", "Afrobeat", "Gospel", "Ska", "Grunge"
]

brani_generi = []

countBrani = 0
TotBrani = 4
TotTrackForGen = (len(genres)/TotBrani)+1
end = False
#Cerco per genere
titles = []
for _,gen in enumerate(genres):
    print(f"{_+1}) Genere: " + gen)
    count = 0
    countTrackGen = 0
    if end:
        break
    #Cerco le playlist su Spotify per ogni genere e conzidero le canzoni all'interno di quel genere 
    for __,res in enumerate( sp.search(q=f"{gen}", type='playlist')['playlists']['items']):
        #print(f"\n\n||\n{res}\n||\n\n")
        if res != None:
            if count == 4:
                break            
            #print("\tNome: "+res['name'])
            #print("\tId: "+res['id'])
            #print("\tDesc: "+res['description'])
            #print("\tLink: "+res['external_urls']['spotify'])
            totCanz = res['tracks']['total']
            #print("\ttotale: "+str(totCanz))
            if totCanz > 100:
                totCanz = 99

            # if totCanz > 20:
            #     totCanz = 20
            count+=1
            playlistTracks = sp.playlist_tracks(res['id'], fields=None, limit=totCanz, offset=0)
            for ___, track in enumerate(playlistTracks['items']):
                #print(___['track']['name'])
                # if countBrani == TotBrani:
                #     end = True
                #     break
                if countTrackGen == TotTrackForGen:
                    end = True
                    break
                if track != None:
                    if track['track'] != None:
                        #print(f"\t\tTrack {___+1}:",track['track']['name'],track['track']['id'])
                        if track['track']['name'] not in titles:
                            if track['track']['artists'][0]['name']:
                                brani_generi.append(f"{gen}:{track['track']['name'].replace(":","").replace("_","").replace(",","")}_{track['track']['artists'][0]['name'].replace(":","").replace("_","").replace(",","")}_{countTrackGen}")
                            #artist = track['track']['artists'][0]['name']
                                countBrani+=1
                                countTrackGen+=1
                                titles.append(track['track']['name'])
                        #print(f"Tot: {countBrani}")
            
        #print(res['owner']['display_name'])
print("\n\n\n\n")
print("Brani Generi:")
for _, brano in enumerate(brani_generi):
    print(f"{_+1})",brano)

print("Totale:", len(brani_generi))

featuresTracks = {}
for _ in range(len(brani_generi)):
    tilte = brani_generi[_].split(":")[1].split("_")[0]
    #print(f"\n\n\n\n{_+1})Download '{tilte}' ...\n\n")
    #d = download_youtube_audio(tilte, "Progetto/Tracks")
    #print(_)
    #print(f"{brani_generi[_].split(":")[1].split("_")[0]},{brani_generi[_].split(':')[1].split('_')[1]},{brani_generi[_].split(':')[1].split('_')[2]}")
    featuresTracks[f'{_}'] = {
        "id_track": _,
        "title": f"{tilte} {brani_generi[_].split(':')[1].split('_')[1]}",
        'artist': brani_generi[_].split(':')[1].split('_')[1],
        "genre": brani_generi[_].split(":")[0],
        "id_track_genre":brani_generi[_].split(':')[1].split('_')[2],
        #"features": d
    }
    #print("Download da YouTube Completato")

#pprint(featuresTracks)

csv = open("Tracks1.csv", "w",encoding="utf-16")
csv.write("id_track,title,artist,genre,genre,id_track_genre\n")
for _, track in enumerate(featuresTracks):
    try:
        csv.write(f"{featuresTracks[track]['id_track']},{featuresTracks[track]['title']},{featuresTracks[track]['artist']},{featuresTracks[track]['genre']},{featuresTracks[track]['id_track_genre']}\n")
    except Exception as e:
        print(f"Errore con {featuresTracks[track]['title']}: {e}")
        # Se c'è un errore, continua a scrivere gli altri brani
        continue

    #csv.write(f"{featuresTracks[track]['genre']},{featuresTracks[track]['features']}\n") 
print("\n\nTotale:", len(brani_generi))
print(f"IN TITLES: {len(titles)}")
print("\nEND")



