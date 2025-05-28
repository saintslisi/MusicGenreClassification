import pandas as pd

# File paths
train_file = "train_balanced.csv"
test_file = "test_clean.csv"
tracks_file = "Tracks1.csv"

# Load datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
tracks_df = pd.read_csv(tracks_file, encoding='utf-16')  

print(tracks_df.columns)

tracks_df = tracks_df[['id_track', 'title']]

# Merge title into train and test sets
train_merged = pd.merge(train_df, tracks_df, on='id_track', how='left')
test_merged = pd.merge(test_df, tracks_df, on='id_track', how='left')

# Create a new textual feature, e.g., title in lowercase
train_merged['title_text'] = train_merged['title'].fillna('').str.lower()
test_merged['title_text'] = test_merged['title'].fillna('').str.lower()

# Save new CSVs
train_merged.to_csv("train_balanced_title.csv", index=False)
test_merged.to_csv("test_clean_title.csv", index=False)