import librosa
import matplotlib.pyplot as plt
import numpy as np 
audio, sr = librosa.load("sample/allenginersrunning.wav")
plt.figure(figsize=(13, 4))
librosa.display.waveshow(audio, sr=sr)
font1 = {"family": "serif", "color": "red", "size": 20}
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Amplitude", fontdict=font1)
plt.title("Sound information ", fontdict=font1)
plt.tight_layout()
plt.show()
fft = np.fft.fft(audio)
fre = np.fft.fftfreq(len(audio))
plt.plot(fft,np.abs(fft))
plt.xlabel("frequency",fontdict=font1)
plt.ylabel("magnitude",fontdict=font1)
plt.title("Dull spectra details")
plt.show()