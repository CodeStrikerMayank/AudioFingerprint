import librosa
import matplotlib.pyplot as plt 
audio ,sr = librosa.load("sample/allenginersrunning.wav")
print(len(audio))
print(sr)