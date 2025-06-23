"""
Script per la ricerca, il download, la conversione e l’estrazione delle feature audio da YouTube.

Funzionalità principali:
- Ricerca del primo video corrispondente a un nome di canzone su YouTube.
- Verifica che il video non superi una durata massima (default 25 minuti).
- Download dell'audio in formato MP3 tramite `yt_dlp` e conversione con FFmpeg.
- Estrazione dei secondi centrali (es. 30 sec) dalla traccia audio.
- Calcolo delle feature audio (MFCC, Chroma, ZCR, Spectral Contrast, ecc.) tramite `extract_features`.

Output:
- Dizionario contenente le feature audio della clip centrale estratta dal brano.

Dipendenze:
- pytube, yt_dlp, pydub, librosa, numpy, FeatureExtractor (modulo esterno locale)
"""

from pytube import Search
import yt_dlp
from pydub import AudioSegment
from Dataset.FeatureExtractor import extract_features


def download_youtube_audio(song_name, path="Demo/Dataset/Tracks", nameFile="tmp", extract_sec=30) -> dict:
    """Scarica l'audio di una canzone da YouTube (max 25 min) ed estrae le feature audio centrali."""

    if nameFile is None:
        nameFile = song_name.replace(" ", "_")

    def search_youtube(query):
        search = Search(query)
        return search.results[0].watch_url if search.results else None

    def check_video_duration(url, max_minutes=25):
        """Controlla che il video non superi max_minutes minuti."""
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                duration = info.get("duration", 0)
                print(f"⏱️ Durata video: {duration // 60} min {duration % 60} sec")
                return duration <= max_minutes * 60
        except Exception as e:
            print(f"⚠️ Errore nel controllo della durata: {e}")
            return False

    def extract_middle_x_seconds(filepath) -> tuple[bool, int]:
        """Estrae i secondi centrali dal file audio e li salva come nuovo MP3."""
        milsec = extract_sec * 1000
        try:
            audio = AudioSegment.from_mp3(filepath)
            duration_ms = len(audio)

            if duration_ms < milsec:
                print(f"⚠️ Il file è più corto di {extract_sec} secondi.")
                return False, 400

            start = max((duration_ms // 2) - (milsec // 2), 0)
            end = start + milsec
            middle = audio[start:end]

            new_filepath = filepath.replace(".mp3", f"{extract_sec}.mp3")
            middle.export(new_filepath, format="mp3")
            return True, 200
        except Exception as e:
            print(f"❌ Errore durante l'estrazione: {e}")
            return False, 400

    def download_audio(youtube_url, filename="song.mp3") -> tuple[bool, int]:
        """Scarica e converte l'audio del video YouTube in MP3 + estrazione clip centrale."""
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

            success, _ = extract_middle_x_seconds(f"{filename}.mp3")
            print("✅ Estrazione clip centrale completata." if success else "❌ Estrazione fallita.")
            return success, 200 if success else 400
        except Exception as e:
            print(f"❌ Errore durante il download: {e}")
            return False, 400

    # Avvio del processo
    youtube_url = search_youtube(song_name)
    print("🔍 YouTube URL trovato:", youtube_url)

    if not youtube_url:
        print("❌ Nessun video trovato per:", song_name)
        return None

    if not check_video_duration(youtube_url):
        print("❌ Brano troppo lungo, download annullato.")
        return None

    full_path = f"{path}/{nameFile}"
    success, _ = download_audio(youtube_url, full_path)

    if success:
        print("✅ Download completato con successo.")
        features = extract_features(f"{path}/tmp{extract_sec}.mp3", sec=extract_sec)
        if features:
            print("✅ Estrazione delle caratteristiche completata.")
            return features
        else:
            print("❌ Estrazione delle caratteristiche fallita.")
    else:
        print("❌ Download fallito.")

    return None
