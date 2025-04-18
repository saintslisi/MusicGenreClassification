from pytube import Search
import yt_dlp
from pydub import AudioSegment
from FeatureExtractor import extract_features

from pprint import pprint

def download_youtube_audio(song_name, path="Progetto/Tracks", nameFile="tmp",extract_sec = 30) -> dict:
    """Scarica l'audio di una canzone da YouTube (max 10 min) e restituisce un dizionario con le feature audio."""

    if nameFile is None:
        nameFile = song_name.replace(" ", "_")

    def search_youtube(query):
        search = Search(query)
        return search.results[0].watch_url if search.results else None

    def check_video_duration(url, max_minutes=25):
        """Controlla che il video non duri più di max_minutes minuti"""
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                duration = info.get("duration", 0)  # in secondi
                print(f"⏱️ Durata video: {duration // 60} min {duration % 60} sec")
                return duration <= max_minutes * 60
        except Exception as e:
            print(f"⚠️ Errore nel controllo della durata: {e}")
            return False

    def extract_middle_x_seconds(filepath) -> tuple[bool, int]:
        milsec = extract_sec * 1000
        try:
            audio = AudioSegment.from_mp3(filepath)
            duration_ms = len(audio)
            if duration_ms < milsec:
                print(f"⚠️ Il file è più corto di {extract_sec} secondi.")
                return False, 400

            start = (duration_ms // 2) - milsec // 2
            if start < 0:
                start = 0
            end = start + milsec
            middle = audio[start:end]

            new_filepath = filepath.replace(".mp3", f"{extract_sec}.mp3")
            middle.export(new_filepath, format="mp3")
            return True, 200
        except Exception as e:
            print(f"❌ Errore durante l'estrazione: {e}")
            return False, 400

    def download_audio(youtube_url, filename="song.mp3") -> tuple[bool, int]:
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": filename,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192"
                }]
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            if extract_middle_x_seconds(f"{filename}.mp3")[0]:
                print(f"✅ Estrazione dei {extract_sec} secondi centrali completata.")
            else:
                print(f"❌ Estrazione dei {extract_sec} secondi centrali fallita.")
            return True, 200
        except Exception as e:
            print(f"❌ Errore durante il download: {e}")
            return False, 400

    # Cerca il video su YouTube
    youtube_url = search_youtube(song_name)
    print("🔍 YouTube URL trovato:", youtube_url)

    if youtube_url:
        max_minutes = 25
        if not check_video_duration(youtube_url, max_minutes):
            print(f"❌ Brano troppo lungo (> {max_minutes} minuti), download annullato.")
            return None

        full_path = f"{path}/{nameFile}"
        if download_audio(youtube_url, full_path)[0]:
            print("✅ Download completato con successo.")
            features = extract_features(f"{path}/tmp{extract_sec}.mp3",sec=extract_sec)
            if features:
                print("✅ Estrazione delle caratteristiche completata.")
                return features
            else:
                print("❌ Estrazione delle caratteristiche fallita.")
                return None
        else:
            print("❌ Download fallito.")
            return None
    else:
        print("❌ Nessun video trovato per:", song_name)
        return None

#download_youtube_audio("Sere Nere Tiziano Ferro", path="Progetto/Tracks", nameFile="Sere_Nere_Tiziano_Ferro")