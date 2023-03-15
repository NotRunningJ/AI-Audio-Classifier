import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math

"""
A class to show visually what is being passed throught the autoencoder & CNN
"""

def mfcc_noise(wav, sr, pad2d):
    RMS = math.sqrt(np.mean(wav**2))
    noise = np.random.normal(0, RMS*0.2, wav.shape[0])
    wav_noise = wav+noise
    hop_length = 512 # in num. of samples
    n_fft = 2048 # window in num. of samples
    mfcc = librosa.feature.mfcc(wav, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    padded_mfcc = pad2d(mfcc,20) #TODO put noise back in
    return padded_mfcc


def mfcc_no_noise(wav, sr, pad2d):
    hop_length = 512 # in num. of samples
    n_fft = 2048 # window in num. of samples
    mfcc = librosa.feature.mfcc(wav, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    padded_mfcc = pad2d(mfcc,20)
    return padded_mfcc
    

pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

file = "audio/9_59_17.wav"

signal, sr = librosa.load(file, sr = 9000)

no_noise = mfcc_no_noise(signal, sr, pad2d)
noise = mfcc_noise(signal, sr, pad2d)


# display MFCC without noise
plt.figure(figsize=(5,3))
librosa.display.specshow(no_noise, sr=sr, hop_length=512)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCC No Noise")
# show the plot
plt.show()

# display MFCC with noise
plt.figure(figsize=(5,3))
librosa.display.specshow(noise, sr=sr, hop_length=512)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCC Noise")
# show the plot
plt.show()


