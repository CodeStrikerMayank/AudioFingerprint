import librosa as li
import numpy as np
import matplotlib.pyplot as pt
from sklearn.metrics.pairwise import cosine_similarity


def main(compare, target=None):
    """ creating a logic to get percntage how much the values get match """
    audio, sr = li.load(compare)
    audio = li.util.fix_length(audio, size=5 * 20500)
    data = li.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    k = np.mean(data,axis=1)
    audio1 , sr1 = li.load(target)
    audio1 = li.util.fix_length(audio1,size=5*20500)
    data1 = li.feature.mfcc(y=audio1,sr=sr1,n_mfcc=13)
    k1 = np.mean(data1,axis=1)
    value = cosine_similarity([k,k1])[0][0]
    return value 

print(main(compare="sample\enginersrunning.wav",target="sample\enginersrunning.wav"))
