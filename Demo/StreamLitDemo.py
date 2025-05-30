import streamlit as st

from count import count_songs_by_genre
from Dataset.MusicDownloader import download_youtube_audio
from Modello.UseModel import *

def reset_new_song_state():
    for key in ["download_done", "audio_path", "x_new", "prediction_result"]:
        st.session_state.pop(key, None)
# -- INIZIO APP --

st.title("🎶 Classificatore di Generi Musicali")

# Stato persistente per scelta input
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None

# Layout dei pulsanti
col1, col2 = st.columns(2)
with col1:
    if st.button("➕ Inserisci nuova canzone"):
        reset_new_song_state()
        st.session_state.input_mode = "nuova"
with col2:
    if st.button("📁 Seleziona canzone esistente"):
        st.session_state.input_mode = "esistente"

# --- NUOVA CANZONE ---
if st.session_state.input_mode == "nuova":
    titolo = st.text_input("Inserisci titolo della canzone", key="titolo")
    artista = st.text_input("Inserisci artista della canzone", key="artista")

    if titolo and artista and "download_done" not in st.session_state:
        st.write("Scarico la canzone, attendi...")
        titoloArtista = f"{titolo} {artista}"
        path = "Demo/Track"
        data = download_youtube_audio(titoloArtista, path, extract_sec=90)
        st.session_state.download_done = True
        st.session_state.audio_path = f"{path}/tmp.mp3"
        st.session_state.x_new = insertData(data, titoloArtista, artista)
        st.success("Download completato.")
    
    if "audio_path" in st.session_state:
        st.audio(st.session_state.audio_path, format="audio/mp3")

    if "x_new" in st.session_state:
        if st.button("🎧 Predici il genere musicale", key="predici_nuova"):
            generi, probabilità,YGeneri = useModel(x=st.session_state.x_new)
            st.session_state.prediction_result = list(zip(generi, probabilità))
            

    if "prediction_result" in st.session_state:
        mess = "Genere corretto: "

        YGenres = ast.literal_eval(YGeneri)
        st.subheader("🎯 Risultati della predizione:")
        for genre, prob in st.session_state.prediction_result:
            st.write(f"- {genre}: **{prob * 100:.2f}%**")

        mess = "Genere corretto: "
        if len(YGenres) < 2:
            mess+=f"**{YGenres[0]}**"
        else:
            for _,g in enumerate(YGenres):
                mess += f"**{g}**"
                if _ < len(YGenres)-1:
                    mess += ", "
        st.write(mess)     

        st.session_state.clear()
        st.session_state.input_mode = None
        st.session_state.download_done = False
        st.session_state.x_new = None
        st.session_state.prediction_result = None
        st.write("Sessione resettata.")      
        titolo = None
        artista = None
        #reset_new_song_state()      
# --- RESET SESSIONE ---


# --- SELEZIONE DA FILE ---
elif st.session_state.input_mode == "esistente":
    fd = pd.read_csv('Demo/Modello/TestTracks.csv', encoding="utf-16")
    dim = fd.shape[0]

    st.write("Scegli una canzone dal dataset:")
    x = st.slider("Indice traccia", 0, dim-1)
    title = fd.iloc[x]['title']
    traccia = f"🎵 '{title.replace(f" {fd.iloc[x]['artist']}","")}' di {fd.iloc[x]['artist']}"
    st.write(traccia)

    if st.button("🎧 Predici il genere musicale"):

        generi,probabilità,YGeneri = useModel(x=x)
        YGenres = ast.literal_eval(YGeneri)
        
        #print(f"\n\nT GENERI\n\n:  {YGenres}")
        st.subheader("🎯 Risultati della predizione:")
        for genre, prob in zip(generi,probabilità):
            st.write(f"- {genre}: **{prob*100:.2f}%**")
        mess = "Genere corretto: "
        if len(YGenres) < 2:
            mess+=f"**{YGenres[0]}**"
        else:
            for _,g in enumerate(YGenres):
                mess += f"**{g}**"
                if _ < len(YGenres)-1:
                    mess += ", "
        st.write(mess)            

if st.button("📊 Vedi distribuzione dei generi"):
    g, t = count_songs_by_genre('Demo/Dataset/train_balanced.csv')
    st.write(f"Totale canzoni nel dataset: **{t}**")
    st.write("Numero di canzoni per genere:")
    st.dataframe(g)

