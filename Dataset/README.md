# Istruzioni Dataset Music Genre Classification

**INSERIRE IL FILE `.env` nella propria directory**

# link al dataset su HuggingFace
https://huggingface.co/datasets/Granataa/MusicClassificator

## Creazione del Dataset

### BuildDataste.py
- Per 30 generi di base, cerca fino a 4 playlist su Spotify per ogni genere.
- Inserisce i titoli delle canzoni e il genere della playlist in un file `Tracks.csv`, creando così una lista di canzoni da poter scaricare.

### DEF_batch.py
- Dato il file `Tracks.csv` con l'elenco di canzoni, viene letto e, se interrotto, riprende dall'ultima canzone non caricata.
- Per ogni canzone, chiama la funzione di `MusicDownloader.py` per scaricare la traccia ed estrarne le feature, che verranno inserite in un nuovo file `TrackFeatures.csv` (contiene id della canzone, genere e features).

### MusicDownloader.py
- Cerca e scarica la traccia audio e un sample di 30 secondi della parte centrale della canzone.
- Chiama la funzione del file `FeatureExtractor.py` che ritorna le feature audio di quella traccia.

### FeatureExtractor.py
- Usa la libreria `librosa` per estrarre le feature della traccia e ritorna un dizionario con quelle feature.

### count.py
- Conta quante canzoni ci sono nel file `Tracks.csv`, per ogni genere e quanti generi ci sono.

### sistemaDataset.py
- Prendendo il dataset originale formato da 29 generi, fa il merge di alcuni generi in 9 macro-generi.

### id-genere.txt
- File con l'id e il nome del macro-genere.

### overSampling.py
- Prende il dataset sistemato da `sistemaDataset.py` e lo bilancia in modo da avere lo stesso numero di canzoni per ogni genere facendo un over-sampling.
- L'output sono due file CSV: un training set bilanciato e un test set pulito senza canzoni duplicate.

### titleOnDataset.py
- Script che legge i file CSV del training set e del test set, unendoli con un file separato contenente gli ID e i titoli dei brani. Aggiunge una nuova colonna al dataset, contenente il titolo del brano in formato testuale (lowercase).
