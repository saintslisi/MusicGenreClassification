import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# === 1. Carica il dataset originale (non ancora sovracampionato) ===
df = pd.read_csv("dataset_macro_generi.csv")

# === 2. Split stratificato in train e test ===
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["macro_genre_id"],
    random_state=42
)

print(f"📊 Train size: {len(train_df)}")
print(f"📊 Test size: {len(test_df)}")

# === 3. Sovracampionamento SOLO sul training set ===
class_counts = train_df["macro_genre"].value_counts()
mean_size = class_counts.max()#int(round(class_counts.mean()))
print(f"\n🎯 Sovracampionamento del training set a {mean_size} campioni per classe...\n")

oversampled_frames = []

for genre, group in train_df.groupby("macro_genre"):
    if len(group) < mean_size:
        oversampled = resample(group, replace=True, n_samples=mean_size, random_state=42)
        print(f"🔁 '{genre}' portato da {len(group)} a {len(oversampled)}")
        oversampled_frames.append(oversampled)
    elif len(group) > mean_size:
        undersampled = group.sample(n=mean_size, random_state=42)
        print(f"🔽 '{genre}' ridotto da {len(group)} a {len(undersampled)}")
        oversampled_frames.append(undersampled)
    else:
        print(f"✅ '{genre}' già a {len(group)} (nessun cambiamento)")
        oversampled_frames.append(group)

# === 4. Unisci e mescola i dati sovracampionati ===
train_balanced_df = pd.concat(oversampled_frames).sample(frac=1, random_state=42).reset_index(drop=True)

# === 5. Salva i dataset ===
train_output = "train_balanced.csv"
test_output = "test_clean.csv"

train_balanced_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

print(f"\n💾 Train bilanciato salvato in: {train_output}")
print(f"💾 Test pulito salvato in: {test_output}")
print(f"\n📊 Train size dopo oversampling: {len(train_balanced_df)}")
print(f"📊 Test size dopo oversampling: {len(test_df)}")