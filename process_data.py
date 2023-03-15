import os
import math
import librosa
import numpy as np
import random as rn
import librosa.display

from keras.utils.np_utils import to_categorical


# create a mfcc with noise in it
def mfcc_noise(wav, sr, pad2d):
    RMS = math.sqrt(np.mean(wav**2))
    noise = np.random.normal(0, RMS * 0.2, wav.shape[0])
    wav_noise = wav+noise
    hop_length = 512 # in num. of samples
    n_fft = 2048 # window in num. of samples
    mfcc = librosa.feature.mfcc(wav_noise, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    padded_mfcc = pad2d(mfcc,20) 
    return padded_mfcc


# create a regular mfcc from audio file
def mfcc_no_noise(wav, sr, pad2d):
    hop_length = 512 # in num. of samples
    n_fft = 1024 # window in num. of samples
    mfcc = librosa.feature.mfcc(wav, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    padded_mfcc = pad2d(mfcc,20)
    return padded_mfcc


# read the data in, providing training, validation, and test datasets with & without noise
def preprocess_data(folders):
    filenames=[]
    num_files = folders*500

    counter = 0 # keeps track of number of folders of data
    for d in os.listdir('data'):
        if counter < folders:
            for file in os.listdir(os.path.join('data', d)):
                filenames.append(os.path.join('data', d, file))
        counter = counter + 1

    rn.shuffle(filenames) # shuffle..important to do it here so our datasets are randomized

    train_ds = []
    train_labels = []
    train_ds_noisy = []
    train_labels_noisy = []
    test_ds = []
    test_labels = []
    test_ds_noisy = []
    test_labels_noisy = []
    val_ds = []
    val_labels = []
    val_ds_noisy = []
    val_labels_noisy = []
    pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

    x = 0 # keeps track of number of .wav files read in so far
    for i in range (int(num_files*0.8)): # 80% of data
        x = i
        temp = filenames[x].split('_')
        str = temp[0]
        length = len(str)
        digit = str[length -1]
        wav, sr = librosa.load(os.path.join(filenames[x]), sr=9000)
        train_ds.append(mfcc_no_noise(wav, sr, pad2d))
        train_ds_noisy.append(mfcc_noise(wav, sr, pad2d))
        train_labels.append(digit)
        train_labels_noisy.append(digit)

    for i in range (int(num_files*0.1)): # 10% of data
        x = x + 1
        temp = filenames[x].split('_')
        str = temp[0]
        length = len(str)
        digit = str[length -1]
        wav, sr = librosa.load(os.path.join(filenames[x]), sr=9000)
        test_ds.append(mfcc_no_noise(wav, sr, pad2d))
        test_ds_noisy.append(mfcc_noise(wav, sr, pad2d))
        test_labels.append(digit)
        test_labels_noisy.append(digit)

    for i in range(int(num_files*0.1)): # 10% of data
        x = x + 1
        temp = filenames[x].split('_')
        str = temp[0]
        length = len(str)
        digit = str[length -1]
        wav, sr = librosa.load(os.path.join(filenames[x]), sr=9000)
        val_ds.append(mfcc_no_noise(wav, sr, pad2d))
        val_ds_noisy.append(mfcc_noise(wav, sr, pad2d))
        val_labels.append(digit)
        val_labels_noisy.append(digit)
    
    train_labels = to_categorical(np.array(train_labels))
    train_labels_noisy = to_categorical(np.array(test_labels_noisy))
    test_labels = to_categorical(np.array(test_labels))
    test_labels_noisy = to_categorical(np.array(test_labels_noisy))
    val_labels = to_categorical(np.array(val_labels))
    val_labels_noisy = to_categorical(np.array(val_labels_noisy))
    train_ds = np.array(train_ds)
    train_ds_noisy = np.array(train_ds_noisy)
    test_ds = np.array(test_ds)
    test_ds_noisy = np.array(test_ds_noisy)
    val_ds = np.array(val_ds)
    val_ds_noisy = np.array(val_ds_noisy)

    train_ds_ex = np.expand_dims(train_ds, -1)
    train_ds_noisy_ex = np.expand_dims(train_ds_noisy, -1)
    test_ds_ex = np.expand_dims(test_ds, -1)
    test_ds_noisy_ex = np.expand_dims(test_ds_noisy, -1)
    val_ds_ex = np.expand_dims(val_ds, -1)
    val_ds_noisy_ex = np.expand_dims(val_ds_noisy, -1)

    return train_ds_ex, train_labels, train_ds_noisy_ex, train_labels_noisy, test_ds_ex, \
        test_labels, test_ds_noisy_ex, test_labels_noisy, val_ds_ex, val_labels, \
            val_ds_noisy_ex, val_labels_noisy




