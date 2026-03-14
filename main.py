import librosa
import matplotlib.pyplot as plt

audio, sr = librosa.load("sample/allenginersrunning.wav")
plt.figure(figsize=(13, 4))
librosa.display.waveshow(audio, sr=sr)
font1 = {"family": "serif", "color": "red", "size": 20}
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Amplitude", fontdict=font1)
plt.title("Sound information ", fontdict=font1)
plt.tight_layout()
plt.show()
