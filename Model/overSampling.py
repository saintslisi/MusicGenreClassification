import pandas as pd
from sklearn.utils import resample

# Carica il dataset già mappato con 'macro_genre' e 'macro_genre_id'
df = pd.read_csv("../Dataset/dataset_macro_generi.csv")

# Trova la dimensione massima tra i macro-generi
max_size = df["macro_genre"].value_counts().max()
print(f"🎯 Sovracampionamento a {max_size} campioni per macro-genere...\n")

# Lista dei gruppi sovracampionati
oversampled_frames = []

# Cicla per ogni macro-genere e sovracampiona
for genre, group in df.groupby("macro_genre"):
    if len(group) < max_size:
        # Sovracampionamento con replacement
        oversampled = resample(group, replace=True, n_samples=max_size, random_state=42)
        print(f"🔁 '{genre}' portato da {len(group)} a {len(oversampled)}")
        oversampled_frames.append(oversampled)
    else:
        print(f"✅ '{genre}' già a {len(group)} (nessun sovracampionamento)")
        oversampled_frames.append(group)

# Unisce tutti i gruppi sovracampionati
df_balanced = pd.concat(oversampled_frames)

# Mescola le righe (opzionale ma consigliato)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Salva il dataset bilanciato
output_file = "../Dataset/dataset_macro_generi_balanced.csv"
df_balanced.to_csv(output_file, index=False)
print(f"\n💾 Dataset bilanciato salvato in: {output_file}")
