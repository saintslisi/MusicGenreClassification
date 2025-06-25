# Music Genre Classification

Modello di Machine Learning che, data in input una traccia audio, riesce a classificarla tra 9 macro-generi diversi.

## Generi possibili

| Codice | Genere             |
|--------|--------------------|
| 0      | Ambient/Other      |
| 1      | Classical          |
| 2      | Electronic         |
| 3      | Folk/Country       |
| 4      | Hip-Hop            |
| 5      | Jazz/Blues         |
| 6      | Pop                |
| 7      | Reggae/Afrobeat    |
| 8      | Rock               |

## Struttura delle cartelle

- **Dataset**: contiene tutti i file utilizzati per creare, gestire e sistemare il dataset.
- **Demo**: contiene i file per la demo Streamlit.
- **Model**: contiene tutti gli script e i file utilizzati per l'addestramento del modello.

La maggior parte dei tataset poichè di grandi dimensioni sono stati caricati su Hugging Face: https://huggingface.co/datasets/Granataa/MusicClassificator
