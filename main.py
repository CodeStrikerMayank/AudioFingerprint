import librosa as lb
import numpy as np
import matplotlib.pyplot as pt
from dtw import dtw


def extract(file):
    audio1, sr = lb.load(file)
    audio = lb.util.normalize(audio1)
    audio = lb.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    data, magnitude = lb.piptrack(y=audio1, sr=sr)
    pitch = []
    for i in range(data.shape[1]):
        index = magnitude[:, i].argmax()
        pitch.append(data[index, i])

    pitch = np.array(pitch)
    pitch[pitch == 0] = np.nan
    pitch = np.nanmean(pitch)
    return audio, pitch


def compare():
    audio, pitch = extract(r"sample\Target hit.mp3")
    audio1, pitch1 = extract(r"sample\Target hit.mp3")
    result = dtw(audio.T, audio1.T)
    distance = result.distance
    pitche = abs(pitch - pitch1)
    return distance, pitche


def decision():
    data, pitch = compare()
    si = 100 / (1 + data)
    print(round(si, 2), "%")
    print(pitch, "hz")


decision()
