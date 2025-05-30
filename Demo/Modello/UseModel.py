from Modello.ModelTraining import *
from Modello.InsertData import *

def useModel(ModelPath = "Demo/Modello/modello_sicuro.pth", ScalerPath = "Demo/Modello/scaler_sicuro.pkl",x = 0):
    """Funzione per usare il modello
        - ModelPath: Path del modello da usare
    """
    scaler = joblib.load(ScalerPath)
    model = MLPClassifier()
    model.load_state_dict(torch.load(ModelPath, map_location=torch.device('cpu')))  
    model.eval()

    # Preprocess
    pathOUT = "Demo/Modello/TestTracksFeatures.csv"
    test_df = pd.read_csv(pathOUT)
    print(f"Preprocesso la riga {x}: {test_df.iloc[x]}")
    vettore = preprocess_row(test_df.iloc[x])
    print("Fine preprocessing")
    vettore_scaled = scaler.transform([vettore])
    tensor = torch.tensor(vettore_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1).numpy()[0]

    label_encoder = LabelEncoder()
    label_encoder.fit(["Ambient/Other","Classical","Electronic","Folk/Country","Hip-Hop","Jazz/Blues","Pop","Reggae/Afrobeat","Rock"])
    # label_encoder.fit(['Reggae', 'Jazz', 'Funk', 'Soul', 'House', 'Dubstep', 'Electronic', 'Blues',
    #                     'Disco', 'Pop', 'Classical', 'Hip-Hop', 'Drum and Bass', 'R&B', 'Techno',
    #                     'Country', 'Folk', 'Punk', 'Metal', 'Trance', 'Rock'])
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    predicted_genres = label_encoder.inverse_transform(top3_indices) 
    print("Ritorno corretto")
    testTrack_df = pd.read_csv("Demo/Modello/TestTracks.csv", encoding="utf-16")
    realGenres = (testTrack_df.iloc[x])["genre"]
    return predicted_genres, probabilities[top3_indices], realGenres

#useModel(x = 16)