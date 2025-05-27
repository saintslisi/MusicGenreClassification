import pandas as pd

print("📥 Caricamento dataset...")
# Mappa dei generi a macro-generi
genre_to_macro = {
    'Rock': 'Rock',
    'Grunge': 'Rock',
    'Metal': 'Rock',
    'Punk': 'Rock',

    'Pop': 'Pop',
    'Disco': 'Pop',
    'R&B': 'Pop',
    'Soul': 'Pop',

    'Hip-Hop': 'Hip-Hop',

    'House': 'Electronic',
    'Trance': 'Electronic',
    'Techno': 'Electronic',
    'Dubstep': 'Electronic',
    'Electronic': 'Electronic',

    'Blues': 'Jazz/Blues',
    'Jazz': 'Jazz/Blues',
    'Gospel': 'Jazz/Blues',

    'Classical': 'Classical',

    'Reggae': 'Reggae/Afrobeat',
    'Afrobeat': 'Reggae/Afrobeat',
    'Ska': 'Reggae/Afrobeat',

    'Folk': 'Folk/Country',
    'Country': 'Folk/Country',
    'Latin': 'Folk/Country',

    'Ambient': 'Ambient/Other',
    'Funk': 'Ambient/Other',
    'Drum and Bass': 'Ambient/Other',
}

# Leggi il dataset originale (senza modificarlo)
df_original = pd.read_csv("TrackFeatures4.csv")
print(f"✅ Righe originali: {len(df_original)}")

# Rimuovi duplicati
df = df_original.drop_duplicates()
print(f"✅ Righe dopo rimozione duplicati: {len(df)}")

# Mappa il genere nel macro-genere
df["macro_genre"] = df["genre"].map(genre_to_macro)

# Trova e stampa i generi non mappati (che verranno eliminati)
non_mappati = df[df["macro_genre"].isna()]["genre"].unique()
print(f"⚠️ Generi non mappati e quindi eliminati: {list(non_mappati)}")

# Elimina righe con macro_genre NaN
df = df.dropna(subset=["macro_genre"])
print(f"✅ Righe dopo rimozione generi non mappati: {len(df)}")

# Mostra la distribuzione finale dei macro-generi
print("\n📊 Distribuzione macro-generi:")
print(df["macro_genre"].value_counts())

# Crea una mappa macro-genere → ID numerico
macro_genres = sorted(df["macro_genre"].unique())  # ordinati alfabeticamente
macro_genre_to_id = {genre: idx for idx, genre in enumerate(macro_genres)}

# Aggiungi una nuova colonna con gli ID
df["macro_genre_id"] = df["macro_genre"].map(macro_genre_to_id)

# Stampa la mappa degli ID
print("\n🔢 Mappa macro-genere → ID:")
for genre, idx in macro_genre_to_id.items():
    print(f"{idx}: {genre}")

# Salva il nuovo dataset in un file CSV separato
output_path = "dataset_macro_generi.csv"
df.to_csv(output_path, index=False)
print(f"\n💾 Dataset salvato in: {output_path}")
