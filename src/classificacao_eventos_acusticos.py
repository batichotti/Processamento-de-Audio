"""
classificacao_eventos_acusticos.py

Este script realiza a classificação de eventos acústicos utilizando extração de características espectrais e um classificador SVM.

Requerimentos:
- numpy
- pandas
- librosa
- scikit-learn
"""

import numpy as np
import pandas as pd
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# Diretório do dataset
DIR_DATASET = "./ec10_subsampled/"

# Parâmetros de análise espectral
TAXA_AMOSTRAGEM = 44100
TAM_JANELA = 1024
PASSO_JANELA = 256
N_MELS = 128
N_MFCC = 30


def extrair_caracteristicas(arquivo_audio,
                            taxa_amostragem=44100,
                            tam_janela=1024,
                            passo_janela=512,
                            mono=True,
                            n_mfcc=20,
                            n_mels=128,
                            silence_threhold_percentile=None):
    """
    Extrai características espectrais e temporais de um arquivo de áudio.
    """
    parametros_fft = {
        'n_fft': tam_janela,
        'hop_length': passo_janela
    }
    # Carrega o arquivo de áudio
    y, sr = librosa.load(arquivo_audio, sr=taxa_amostragem, mono=mono)

    # Calcula o Mel Espectrograma
    MS = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, **parametros_fft)

    # Características espectrais
    centroid = librosa.feature.spectral_centroid(S=MS, **parametros_fft)
    rolloff = librosa.feature.spectral_rolloff(S=MS, roll_percent=0.85, **parametros_fft)
    flatness = librosa.feature.spectral_flatness(S=MS, **parametros_fft)
    contrast = librosa.feature.spectral_contrast(S=MS, **parametros_fft)
    rms = librosa.feature.rms(y=y, frame_length=tam_janela, hop_length=passo_janela)

    if silence_threhold_percentile is not None:
        # Elimina frames silenciosos com base no RMS
        percentil = np.percentile(rms, silence_threhold_percentile)
        frames_interesse = rms > percentil
        MS = MS[:, frames_interesse.flatten()]

    # MFCCs
    mfccs = librosa.feature.mfcc(S=MS, n_mfcc=n_mfcc)

    # Taxa de cruzamento por zero
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=tam_janela,
                                             hop_length=passo_janela)

    # Médias e desvios padrão das características
    medias = list(map(np.mean, [centroid, rolloff, flatness, contrast, zcr, rms]))
    stds = list(map(np.std, [centroid, rolloff, flatness, contrast, zcr, rms]))

    # Médias e desvios padrão dos MFCCs
    medias_mfcc = np.mean(mfccs, axis=1)
    stds_mfcc = np.std(mfccs, axis=1)

    # Combina todas as características em um vetor
    features = np.concatenate([medias, medias_mfcc, stds, stds_mfcc])
    return features


def extrair_caracteristicas_lista(lista_arquivos, caminho, params_caracteristicas):
    """
    Extrai características de uma lista de arquivos de áudio.
    """
    X = []
    for arquivo in lista_arquivos:
        caracteristicas = extrair_caracteristicas(caminho + arquivo, **params_caracteristicas)
        X.append(caracteristicas)
    return np.array(X)


def main():
    # Parâmetros para extração de características
    params_caracteristicas = {
        'taxa_amostragem': TAXA_AMOSTRAGEM,
        'tam_janela': TAM_JANELA,
        'passo_janela': PASSO_JANELA,
        'n_mels': N_MELS,
        'n_mfcc': N_MFCC,
        'silence_threhold_percentile': 10
    }

    # Carrega o CSV de treinamento
    csv_treino = pd.read_csv(DIR_DATASET + "treino.csv")
    X_treino = extrair_caracteristicas_lista(csv_treino['filename'], DIR_DATASET + "treino/", params_caracteristicas)
    y_treino = csv_treino['category'].values

    # Seleção de características
    vt = VarianceThreshold(threshold=0)
    X_treino = vt.fit_transform(X_treino)
    skb = SelectKBest(f_classif, k=10)
    X_treino = skb.fit_transform(X_treino, y_treino)

    # Pipeline de classificação
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    param_svc = {
        'svc__C': [0.1, 1, 10, 100, 1000],
        'svc__kernel': ['rbf'],
        'svc__gamma': ['scale', 'auto']
    }
    classificador = GridSearchCV(pipeline, param_svc, cv=3)
    classificador.fit(X_treino, y_treino)

    # Carrega o CSV de teste
    csv_teste = pd.read_csv(DIR_DATASET + "teste.csv")
    X_teste = extrair_caracteristicas_lista(csv_teste['filename'], DIR_DATASET + "teste/", params_caracteristicas)
    X_teste = skb.transform(vt.transform(X_teste))
    y_teste = csv_teste['category']

    # Predição e avaliação
    y_pred = classificador.predict(X_teste)
    print(classification_report(y_teste, y_pred))
    ConfusionMatrixDisplay.from_estimator(classificador, X_teste, y_teste,
                                          cmap='Blues', xticks_rotation='vertical')

if __name__ == "__main__":
    main()
