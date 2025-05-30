import numpy as np
import ast
# Preprocessing
def flatten_feature_matrix(matrix):
    return np.array(matrix).flatten()

def preprocess_row(row):
    try:
        print("Preprocessing in corso...")
        mfccs = ast.literal_eval(row['mfccs'])
        print("MFCCs preprocessati.")
        chroma = ast.literal_eval(row['chroma'])
        print("Chroma preprocessato.")
        spec_contrast = ast.literal_eval(row['spec_contrast'])
        print("Spectral Contrast preprocessato.")
         # Gestione Zero Crossing Rate e Beats
        zcr = ast.literal_eval(row['zcr'])
        print("Zero Crossing Rate preprocessato.")
        beats = ast.literal_eval(row['beats'])
        print("Beats preprocessati.")
        tempo_feature = float(row['tempo'].replace("[", "").replace("]", ""))
        print("Tempo preprocessato.")

        mfccs_vec = np.concatenate([
            flatten_feature_matrix(mfccs['mean']),
            flatten_feature_matrix(mfccs['std'])
        ])
        print("MFCCs preprocessati.")

        chroma_vec = np.concatenate([
            flatten_feature_matrix(chroma['mean']),
            flatten_feature_matrix(chroma['std'])
        ])
        print("Chroma preprocessato.")
        spec_vec = np.concatenate([
            flatten_feature_matrix(spec_contrast['mean']),
            flatten_feature_matrix(spec_contrast['std'])
        ])
        print("Spectral Contrast preprocessato.")
        zcr_vec = np.concatenate([
            flatten_feature_matrix(zcr['mean']),
            flatten_feature_matrix(zcr['std'])
        ])
        print("Zero Crossing Rate preprocessato.")
        beats_vec = np.array([beats['count'], beats['interval_mean'], beats['interval_std']])
        print("Beats preprocessati.")
        print("Tempo preprocessato.")
        return np.concatenate([mfccs_vec, chroma_vec, spec_vec, zcr_vec, [tempo_feature], beats_vec])
    except Exception as e:
        print(f"Errore nel preprocessing: {e}")
        return np.zeros(5944)
