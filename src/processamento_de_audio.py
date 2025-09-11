"""
processamento_de_audio.py

Este script demonstra operações básicas de processamento de áudio usando a biblioteca librosa e numpy.
Inclui carregamento, normalização, visualização, síntese de tons e resampling.

Requerimentos:
- numpy
- librosa
- IPython
"""

import librosa
import librosa.display
from IPython.display import Audio, display
import numpy as np

# Carrega um exemplo de áudio
x, sr = librosa.load(librosa.example("brahms"), sr=None)
print("Vetor de amostras:", x)
print("Taxa de amostragem:", sr)
print("Quantidade de amostras:", len(x))

duracao = len(x) / sr
print("Duração (s):", duracao)

display(Audio(x, rate=sr))

# Visualiza uma pequena janela do áudio
librosa.display.waveshow(x[sr*10:sr*10+(sr//200)], sr=sr)

# Estatísticas do sinal
print("Valor mínimo e máximo:", np.min(x), np.max(x))

# Normalização
x_norm = x / np.max(np.abs(x))
print("Normalizado - min/max:", np.min(x_norm), np.max(x_norm))

# Conversão para inteiro sem sinal (8 bits)
x_norm = (x_norm + 1)
x8 = (x_norm * 255/2).astype(np.uint8)
print("x8 - min/max:", np.min(x8), np.max(x8))

display(Audio(x, rate=sr))
display(Audio(x8, rate=sr))

# Síntese de tons puros
sr440 = 44100
duracao440 = 2.0 # segundos
t = np.linspace(0, duracao440, int(sr * duracao440), endpoint=False)
tom440 = np.sin(2 * np.pi * 440 * t)
tom880 = np.sin(2 * np.pi * 880 * t)

display(Audio(tom440, rate=sr))
display(Audio(tom880, rate=sr))

# Combinação de tons
som_440_880 = tom440 + tom880
som_440_880_05 = tom440 + 0.5 * tom880
display(Audio(tom440, rate=sr))
display(Audio(tom880, rate=sr))
display(Audio(som_440_880, rate=sr))
display(Audio(som_440_880_05, rate=sr))

# Síntese de acorde
som_do = np.sin(2 * np.pi * 261.63 * t)
som_mi_b = np.sin(2 * np.pi * 312 * t)
som_sol = np.sin(2 * np.pi * 392 * t)
som_acorde_do = som_do + som_mi_b + som_sol
display(Audio(som_acorde_do, rate=sr))

# Resampling do áudio
x, sr = librosa.load(librosa.example("brahms"), sr=None)
x_8k = librosa.resample(x, orig_sr=sr, target_sr=8000)
x_4k = librosa.resample(x, orig_sr=sr, target_sr=4000)
x_3k = librosa.resample(x, orig_sr=sr, target_sr=3000)

display(Audio(x, rate=sr))
display(Audio(x_8k, rate=8000))
display(Audio(x_4k, rate=4000))
display(Audio(x_3k, rate=3000))

print("Comprimento original e resample:", len(x), len(x_3k))
